from __future__ import annotations

import random
import weakref
import threading
from typing import Any, Callable, ClassVar, Iterator, List, Tuple
import pylinalg as la
from time import perf_counter_ns

import numpy as np

from ..resources import Buffer, Texture
from ..utils import array_from_shadertype, logger
from ..utils.trackable import Trackable
from ..utils.bounds import Bounds
from ._events import EventTarget
from ..utils.transform import (
    AffineTransform,
    RecursiveTransform,
)
from ..utils.enums import RenderMask
from ..geometries import Geometry
from ..materials import Material


class IdProvider:
    """Object for internal use to manage world object id's."""

    def __init__(self):
        self._ids_in_use = set([0])
        self._map = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

    def claim_id(self, wobject: WorldObject) -> int:
        """Used by wobjects to claim an id."""
        # We don't simply count up, but keep a pool of ids. This is
        # because an application *could* create and discard objects at
        # a high rate, so we want to be able to re-use these ids.
        #
        # Some numbers:
        # * 4_294_967_296 (2**32) max number for u32
        # * 2_147_483_647 (2**31 -1) max number for i32.
        # *    16_777_216 max integer that can be stored exactly in f32
        # *     4_000_000 max integer that survives being passed as a varying (in my tests)
        # *     1_048_575 is ~1M is 2**20 seems like a good max scene objects.
        # *    67_108_864 is ~50M is 2**26 seems like a good max vertex count.
        #                 which leaves 64-20-26=18 bits for any other picking info.

        # Max allowed id, inclusive
        id_max = 1_048_575  # 2*20-1

        # The max number of ids. This is a bit less to avoid choking
        # when there are few free id's left.
        max_items = 1_000_000

        with self._lock:
            if len(self._ids_in_use) >= max_items:
                raise RuntimeError("Max number of objects reached.")
            id = 0
            while id in self._ids_in_use:
                id = random.randint(1, id_max)
            self._ids_in_use.add(id)
            self._map[id] = wobject

        return id

    def release_id(self, wobject: WorldObject, id: int) -> None:
        """Release an id associated with a wobject."""
        if id > 0:
            with self._lock:
                self._ids_in_use.discard(id)
                self._map.pop(id, None)

    def get_object_from_id(self, id: int) -> WorldObject | None:
        """Return the wobject associated with an id, or None."""
        return self._map.get(id)


id_provider = IdProvider()


class WorldObject(EventTarget, Trackable):
    """Base class for objects.

    This class represents objects in the world, i.e., the scene graph.Each
    WorldObject has geometry to define it's data, and material to define its
    appearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the
    world.

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object. See the documentation
        on the different WorldObject subclasses for what attributes the
        geometry should and may have.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : float
        Value that helps controls the order in which objects are rendered.
    render_mask : str | RenderMask
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    name : str
        The name of the object.

    Notes
    -----
    Use :class:`Group` to collect multiple world objects into a single empty
    world object.

    See Also
    --------
    pygfx.utils.transform.AffineBase
        Various getters and setters defined on ``obj.local`` and ``obj.world``.
    pygfx.utils.transform.AffineTransform
        The class used to implement ``obj.local``.
    pygfx.utils.transform.RecursiveTransform
        The class used to implement ``obj.world``.

    """

    _FORWARD_IS_MINUS_Z = False  # Default is +Z (lights and cameras use -Z)

    _id = 0

    # The uniform type describes the structured info for this object, which represents
    # every "property" that a renderer would need to know in order to visualize it.
    # Put larger items first for alignment, also note that host-sharable structs
    # align at power-of-two only, so e.g. vec3 needs padding.
    # todo: rename uniform to info or something?

    uniform_type: ClassVar[dict[str, str]] = dict(
        world_transform="4x4xf4",
        world_transform_inv="4x4xf4",
        id="i4",
    )

    def __init__(
        self,
        geometry: Geometry | None = None,
        material: Material | None = None,
        *,
        visible: bool = True,
        render_order: float = 0,
        render_mask: str | int = "auto",
        name: str = "",
    ) -> None:
        super().__init__()
        self._parent: weakref.ReferenceType[WorldObject] | None = None

        #: Subtrees of the scene graph that depend on this object.
        self._children: List[WorldObject] = []

        self.geometry = geometry
        self.material = material

        self.name = name

        # Compose complete uniform type
        buffer = Buffer(array_from_shadertype(self.uniform_type), force_contiguous=True)
        buffer.data["world_transform"] = np.eye(4)
        buffer.data["world_transform_inv"] = np.eye(4)

        self._world_last_modified = perf_counter_ns()

        #: The object's transform expressed in parent space.
        self.local = AffineTransform(is_camera_space=self._FORWARD_IS_MINUS_Z)
        #: The object's transform expressed in world space.
        self.world = RecursiveTransform(
            self.local, is_camera_space=self._FORWARD_IS_MINUS_Z, reference_up=(0, 1, 0)
        )

        # Set id
        self._id = id_provider.claim_id(self)
        buffer.data["id"] = self._id

        # Bounds
        self._bounds_geometry = None
        self._bounds_geometry_rev = 0

        #: The GPU data of this WorldObject.
        self.uniform_buffer = buffer

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.render_mask = render_mask
        self.cast_shadow = False
        self.receive_shadow = False

        self.name = name

    def _update_object(self):
        """This gets called (by the renderer) right before being drawn. Good time for lazy updates."""
        world_last_modified = self.world.last_modified
        if world_last_modified > self._world_last_modified:
            self._world_last_modified = world_last_modified
            self._update_world_transform()

    def _update_world_transform(self):
        """This gets called right before being drawn, when the world transform has changed."""
        orig_err_setting = np.seterr(under="ignore")
        self.uniform_buffer.data["world_transform"] = self.world.matrix.T
        self.uniform_buffer.data["world_transform_inv"] = self.world.inverse_matrix.T
        self.uniform_buffer.update_full()
        np.seterr(**orig_err_setting)

    def __repr__(self):
        return f"<pygfx.{self.__class__.__name__} {self.name} at {hex(id(self))}>"

    def __del__(self):
        id_provider.release_id(self, self.id)
        self.local._set_wrapper(
            None
        )  # break the circular reference so GC has it a little easier

    @property
    def up(self) -> np.ndarray:
        """
        Relic of old WorldObjects that aliases with the new ``transform.up``
        direction. Prefer `obj.world.reference_up` instead.

        """

        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        return self.world.reference_up

    @up.setter
    def up(self, value: np.ndarray) -> None:
        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        self.world.reference_up = np.asarray(value)

    @property
    def id(self) -> int:
        """An integer id smaller than 2**31 (read-only)."""
        return self._id

    @property
    def visible(self) -> bool:
        """Whether is object is rendered or not. Default True."""
        return self._store.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._store.visible = bool(visible)

    @property
    def render_order(self) -> float:
        """A number that helps control the order in which objects are rendered.
        Objects with higher ``render_order`` get rendered later.
        Default 0. Also see ``Renderer.sort_objects``.
        """
        return self._store.render_order

    @render_order.setter
    def render_order(self, value: float) -> None:
        self._store.render_order = float(value)

    @property
    def render_mask(self) -> int:
        """Indicates in what render passes to render this object:

        See :obj:`pygfx.utils.enums.RenderMask`:

        If "auto" (the default), the renderer attempts to determine
        whether all fragments will be either opaque or all transparent,
        and only apply the needed render passes. If this cannot be
        determined, it falls back to "all".

        Some objects may contain both transparent and opaque fragments,
        and should be rendered in all passes - the object's contribution
        to each pass is determined on a per-fragment basis.

        For clarity, rendering objects in all passes even though they
        are fully opaque/transparent yields correct results and is
        generally the "safest" option. The only cost is performance.
        Rendering transparent fragments in the opaque pass makes them
        invisible. Rendering opaque fragments in the transparent pass
        blends them as if they are transparent with an alpha of 1.
        """
        return self._store.render_mask

    @render_mask.setter
    def render_mask(self, value: int | str) -> None:
        if value is None:
            value = 0
        if isinstance(value, int):
            pass
        elif isinstance(value, str):
            if value not in dir(RenderMask):
                msg = f"WorldObject.render_mask must be one of {dir(RenderMask)} not {value!r}"
                raise ValueError(msg) from None
            value = RenderMask[value]
        else:
            raise TypeError(
                f"WorldObject.render_mask must be int or str, not {type(value)}"
            )
        # Store the value as an int, because this is a flag, but also for backwards compat.
        self._store.render_mask = value

    @property
    def geometry(self) -> Geometry | None:
        """The object's geometry, the data that defines (the shape of) this object."""
        return self._store.geometry

    @geometry.setter
    def geometry(self, geometry: Geometry | None):
        if not (geometry is None or isinstance(geometry, Geometry)):
            raise TypeError(
                f"WorldObject.geometry must be a Geometry object or None, not {geometry!r}"
            )
        self._store.geometry = geometry

    @property
    def material(self) -> Material | None:
        """The object's material, the data that defines the appearance of this object."""
        # In contrast to the geometry, the material is not stored on self._store,
        # because it should not be tracked, because pipeline-containers are unique
        # for each combi of (wobject, material, renderstate).
        return self._material

    @material.setter
    def material(self, material: Material | None) -> None:
        if not (material is None or isinstance(material, Material)):
            raise TypeError(
                f"WorldObject.geometry must be a Geometry object or None, not {material!r}"
            )
        self._material = material

    @property
    def cast_shadow(self) -> bool:
        """Whether this object casts shadows, i.e. whether it is rendered into
        a shadow map. Default False."""
        return self._cast_shadow  # does not affect any shaders

    @cast_shadow.setter
    def cast_shadow(self, value: bool) -> None:
        self._cast_shadow = bool(value)

    @property
    def receive_shadow(self) -> bool:
        """Whether this object receives shadows. Default False."""
        return self._store.receive_shadow

    @receive_shadow.setter
    def receive_shadow(self, value: bool) -> None:
        self._store.receive_shadow = bool(value)

    @property
    def parent(self) -> WorldObject | None:
        """Object's parent in the scene graph (read-only).
        An object can have at most one parent.
        """
        if self._parent is None:
            return None
        else:
            return self._parent()

    @property
    def children(self) -> Tuple[WorldObject, ...]:
        """tuple of children of this object. (read-only)"""
        return tuple(self._children)

    def add(
        self,
        *objects: WorldObject,
        before: WorldObject | None = None,
        keep_world_matrix: bool = False,
    ) -> WorldObject:
        """Add child objects.

        Any number of objects may be added. Any current parent on an object
        passed in here will be removed, since an object can have at most one
        parent. If ``before`` argument is given, then the items are inserted
        before the given element.

        Parameters
        ----------
        *objects : WorldObject
            The world objects to add as children.
        before : WorldObject
            If not None, insert the objects before this child object.
        keep_world_matrix : bool
            If True, the child will keep it's world transform. It moves in the
            scene graph but will visually remain in the same place. If False,
            the child will keep it's parent transform.

        """
        for obj in objects:
            if obj.parent is not None:
                obj.parent.remove(obj, keep_world_matrix=keep_world_matrix)

            if before is not None:
                idx = self._children.index(before)
            else:
                idx = len(self._children)

            if keep_world_matrix:
                transform_matrix = obj.world.matrix

            obj._parent = weakref.ref(self)
            obj.world.parent = self.world
            self._children.insert(idx, obj)
            self.world.children.insert(idx, obj.world)

            if keep_world_matrix:
                obj.world.matrix = transform_matrix

        return self

    def remove(self, *objects: WorldObject, keep_world_matrix: bool = False) -> None:
        """Removes object as child of this object. Any number of objects may be removed."""
        for obj in objects:
            try:
                self._children.remove(obj)
                self.world.children.remove(obj.world)
            except ValueError:
                logger.warning("Attempting to remove object that was not a child.")
                continue
            else:
                obj._reset_parent(keep_world_matrix=keep_world_matrix)

    def clear(self, *, keep_world_matrix: bool = False) -> None:
        """Removes all children."""

        for child in self._children:
            child._reset_parent(keep_world_matrix=keep_world_matrix)

        self._children.clear()
        self.world.children.clear()

    def _reset_parent(self, *, keep_world_matrix=False):
        """Sets the parent to None.

        xref: https://github.com/pygfx/pygfx/pull/482#discussion_r1135670771
        """

        if keep_world_matrix:
            transform_matrix = self.world.matrix

        self._parent = None
        self.world.parent = None

        if keep_world_matrix:
            self.world.matrix = transform_matrix

    def traverse(
        self, callback: Callable[[WorldObject], Any], skip_invisible: bool = False
    ):
        """Executes the callback on this object and all descendants.

        If ``skip_invisible`` is given and True, objects whose
        ``visible`` property is False - and their children - are
        skipped. Note that modifying the scene graph inside the callback
        is discouraged.
        """

        for child in self.iter(skip_invisible=skip_invisible):
            callback(child)

    def iter(
        self,
        filter_fn: Callable[[WorldObject], bool] | None = None,
        skip_invisible: bool = False,
    ) -> Iterator[WorldObject]:
        """Create a generator that iterates over this objects and its children.
        If ``filter_fn`` is given, only objects for which it returns ``True``
        are included.
        """
        if skip_invisible and not self.visible:
            return

        if filter_fn is None:
            yield self
        elif filter_fn(self):
            yield self

        for child in self._children:
            yield from child.iter(filter_fn, skip_invisible)

    def _get_bounds_from_geometry(self):
        geometry = self.geometry
        if geometry is None:
            self._bounds_geometry = None
        elif isinstance(positions_buf := getattr(geometry, "positions", None), Buffer):
            if self._bounds_geometry_rev == positions_buf.rev:
                return self._bounds_geometry
            self._bounds_geometry = None
            # Get array and check expected shape
            positions_array = positions_buf.data
            if (
                positions_array is not None
                and positions_array.ndim == 2
                and positions_array.shape[1] in (2, 3)
            ):
                self._bounds_geometry = Bounds.from_points(positions_array)
                self._bounds_geometry_rev = positions_buf.rev
        elif isinstance(grid_buf := getattr(geometry, "grid", None), Texture):
            if self._bounds_geometry_rev == grid_buf.rev:
                return self._bounds_geometry
            # account for multi-channel image data
            grid_shape = tuple(reversed(grid_buf.size[: grid_buf.dim]))
            # create aabb in index/data space
            aabb = np.array([np.zeros_like(grid_shape), grid_shape[::-1]], dtype="f8")
            # convert to local image space by aligning
            # center of voxel index (0, 0, 0) with origin (0, 0, 0)
            aabb -= 0.5
            # ensure coordinates are 3D
            # NOTE: important we do this last, we don't want to apply
            # the -0.5 offset to the z-coordinate of 2D images
            if aabb.shape[1] == 2:
                aabb = np.hstack([aabb, [[0], [0]]])
            self._bounds_geometry = Bounds(aabb, None)
            self._bounds_geometry_rev = grid_buf.rev
        else:
            self._bounds_geometry = None
        return self._bounds_geometry

    def get_geometry_bounding_box(self) -> np.ndarray | None:
        bounds = self._get_bounds_from_geometry()
        if bounds is not None:
            return bounds.aabb

    def get_bounding_box(self) -> np.ndarray | None:
        """Axis-aligned bounding box in parent space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            An axis-aligned bounding box, or None when the object does
            not take up a particular space.
        """

        # Collect bounding boxes
        _aabbs = []
        for child in self._children:
            aabb = child.get_bounding_box()
            if aabb is not None:
                trafo = child.local.matrix
                _aabbs.append(la.aabb_transform(aabb, trafo))
        bounds = self._get_bounds_from_geometry()
        if bounds is not None:
            _aabbs.append(bounds.aabb)

        # Combine
        if _aabbs:
            aabbs = np.stack(_aabbs)
            final_aabb = np.zeros((2, 3), dtype=float)
            final_aabb[0] = np.min(aabbs[:, 0, :], axis=0)
            final_aabb[1] = np.max(aabbs[:, 1, :], axis=0)
        else:
            final_aabb = None

        return final_aabb

    def get_bounding_sphere(self) -> np.ndarray | None:
        """Bounding Sphere in parent space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        # NOTE: this currently does not even use the sphere-data from the geometry!
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def get_world_bounding_box(self) -> np.ndarray | None:
        """Axis aligned bounding box in world space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            The transformed axis-aligned bounding box, or None when the
            object does not take up a particular space.

        """
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_transform(aabb, self.world.matrix)

    def get_world_bounding_sphere(self) -> np.ndarray | None:
        """Bounding Sphere in world space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        aabb = self.get_world_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def _wgpu_get_pick_info(self, pick_value) -> dict:
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)

    def look_at(self, target: WorldObject) -> None:
        """Orient the object so it looks at the given position.

        This sets the object's rotation such that its ``forward`` direction
        points towards ``target`` (given in world space). This rotation takes
        reference_up into account, i.e., the rotation is chosen in such a way that a
        camera looking ``forward`` follows the rotation of a human head looking
        around without tilting the head sideways.

        """

        self.world.forward = target - self.world.position
