import pylinalg as la
import numpy as np

from ._base import WorldObject
from ..resources import Buffer
from ..utils import unpack_bitfield, array_from_shadertype
from ..utils.transform import AffineBase, callback
from ..materials import BackgroundMaterial


class Group(WorldObject):
    """A group of objects.

    A Group is useful when manipulating the scene graph as children can be
    jointly moved/scaled/rotated. It has no visual properties.

    Parameters
    ----------
    visible : bool
        If true, the object and its children are visible.

    name : str
        The name of the group.

    """

    def __init__(self, *, visible=True, name=""):
        super().__init__(visible=visible, name=name)


class Scene(Group):
    """Root of the scene graph.

    The scene holds scene-level information (background color, fog, environment
    map) as well as all objects that take part in the rendering process as
    either direct or indirect children/nested objects.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Background(WorldObject):
    """The scene's background.

    Can be e.g. a gradient, a static image or a skybox.

    Parameters
    ----------
    geometry : Geometry
        Must be ``None``. Exists for compliance with the generic WorldObject
        API.
    material : Material
        The material to use when rendering the background.
    kwargs : Any
        Additional kwargs are forwarded to the object's :class:`base class
        <pygfx.objects.WorldObject>`.

    """

    def __init__(self, geometry=None, material=None, render_mask="opaque", **kwargs):
        if geometry is not None and material is None:
            raise TypeError("You need to instantiate using Background(None, material)")
        super().__init__(None, material, render_mask=render_mask, **kwargs)

    @classmethod
    def from_color(cls, *colors):
        """Create a background with a :class:`.BackgroundMaterial`,
        using 1 uniform color, 2 colors for a vertical gradient, or 4 colors (one for each corner).
        """
        return cls(None, BackgroundMaterial(*colors))


class Grid(WorldObject):
    """A grid to help interpret spatial distances.

    The grid by default occupies 1x1 square in the xz plane. It can be scaled,
    rotated, and translated to move it into position. If the grid is infinite
    (``material.infinite``) then scale and in-plane translations are ignored.

    Parameters
    ----------
    geometry : Geometry
        Must be ``None``. Exists for compliance with the generic WorldObject API.
    material : Material
        The material to use when rendering the background.
    orientation : str
        The (initial) grid rotation. Must be 'xy' (default), 'xz', or 'yz'.
        Simply rotates the object, e.g. for 'xz' will do ``self.local.euler_x = np.pi/2``.
    kwargs : Any
        Additional kwargs are forwarded to the object's :class:`base class
        <pygfx.objects.WorldObject>`.
    """

    def __init__(self, geometry=None, material=None, *, orientation, **kwargs):
        if geometry is not None and material is None:
            raise TypeError("You need to instantiate using Grid(None, material)")
        super().__init__(None, material, **kwargs)
        if orientation is not None:
            if orientation == "xy":
                pass
            elif orientation == "xz":
                self.local.euler_x = np.pi / 2
            elif orientation == "yz":
                self.local.euler_y = -np.pi / 2
            else:
                raise ValueError(
                    f"Invalid grid orientation: '{orientation}', must be 'xz', 'xy', or 'yz'."
                )


class Line(WorldObject):
    """An object representing a line using a list of vertices (3D positions).

    Some materials will render the line as a continuous line, while other materials
    will consider each pair of consecutive points a segment.

    The picking info of a Line (the result of ``renderer.get_pick_info()``) will
    for most materials include ``vertex_index`` (int) and ``segment_coord``
    (float, sub-segment coordinate).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """


class Points(WorldObject):
    """A point cloud.

    An object consisting of points represented by vertices (3D positions).

    The picking info of a Points object (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (int) and ``point_coord`` (tuple of 2 float coordinates in
    logical pixels).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """


class Mesh(WorldObject):
    """A mesh.

    An object consisting of triangular faces represented by a set of vertices
    (3D positions) and a set of vertex indices indicating which vertex triplets
    form mesh triangles.

    The picking info of a Mesh (the result of ``renderer.get_pick_info()``) will
    for most materials include ``instance_index`` (int), ``face_index`` (int),
    and ``face_coord`` (tuple of 3 floats). The latter are the barycentric
    coordinates for each vertex of the face (with values 0..1).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def __init__(self, geometry=None, material=None, *args, **kwargs):
        super().__init__(geometry, material, *args, **kwargs)
        self._morph_target_influences = None
        self._morph_target_names = []

    @property
    def morph_target_influences(self):
        """
        An ndarray of weights typically from 0-1 that specify how much of the morph is applied.

        Note:
        When using this attribute, its geometry needs to have the relevant attributes of morph targets.
        """
        return self._morph_target_influences.data["influence"][:-1]

    @morph_target_influences.setter
    def morph_target_influences(self, value):
        morph_attrs = (
            getattr(self.geometry, "morph_positions", None)
            or getattr(self.geometry, "morph_normals", None)
            or getattr(self.geometry, "morph_colors", None)
            or []
        )

        if not morph_attrs:
            return

        morph_target_count = len(morph_attrs)

        assert len(value) == morph_target_count, (
            f"Length of morph target influences must match the number of morph targets. "
            f"Expected {morph_target_count}, got {len(value)}."
        )

        if (
            self._morph_target_influences is None
            or self._morph_target_influences.nitems != morph_target_count + 1
        ):
            self._morph_target_influences = Buffer(
                array_from_shadertype(
                    {
                        "influence": "f4",
                    },
                    morph_target_count + 1,
                )
            )

        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)

        if getattr(self.geometry, "morph_targets_relative", False):
            base_influence = 1.0
        else:
            base_influence = 1 - value.sum()

        # Add the base influence to the end of the array
        value = np.concatenate([value, [base_influence]], axis=0)

        self._morph_target_influences.data["influence"] = value
        self._morph_target_influences.update_range()

    @property
    def morph_target_names(self):
        """
        A list of names for the morph targets.
        """
        return self._morph_target_names

    def _wgpu_get_pick_info(self, pick_value):
        values = unpack_bitfield(
            pick_value, wobject_id=20, index=26, coord1=6, coord2=6, coord3=6
        )
        face_index = values["index"]
        face_coord = [
            values["coord1"] / 63,
            values["coord2"] / 63,
            values["coord3"] / 63,
        ]
        if (
            self.geometry.indices.data is not None
            and self.geometry.indices.data.shape[-1] == 4
        ):
            triangle_index = face_index % 2
            face_index = face_index // 2
            if triangle_index == 0:
                # The sub indices are 0, 1, 2, so we just add the zero for index 3.
                face_coord.append(0.0)
            else:
                # The sub indices are 0, 2, 3. The index 3 uses the
                # face_coord slot of index 1, (see meshshader.py), so
                # we put that at the end and put a zero in its place.
                face_coord = face_coord[0], 0.0, face_coord[2], face_coord[1]

        return {"face_index": face_index, "face_coord": tuple(face_coord)}


class Image(WorldObject):
    """A 2D image.

    The geometry for this object consists only of ``geometry.grid``: a
    texture with the 2D data.

    If no colormap is applied to the material, the data are interpreted as
    colors in sRGB space. To use physical space instead, set the texture's
    colorspace property to ``"physical"``.

    The picking info of an Image (the result of ``renderer.get_pick_info()``)
    will for most materials include ``index`` (tuple of 2 int), and
    ``pixel_coord`` (tuple of float subpixel coordinates).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, x=22, y=22)
        x = values["x"] / 4194303 * tex.size[0] - 0.5
        y = values["y"] / 4194303 * tex.size[1] - 0.5
        ix, iy = int(x + 0.5), int(y + 0.5)
        return {
            "index": (ix, iy),
            "pixel_coord": (x - ix, y - iy),
        }


class Volume(WorldObject):
    """A 3D image.

    The geometry for this object consists only of ``geometry.grid``: a texture
    with the 3D data.

    The picking info of a Volume (the result of ``renderer.get_pick_info()``)
    will for most materials include ``index`` (tuple of 3 int), and
    ``voxel_coord`` (tuple of float subpixel coordinates).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, x=14, y=14, z=14)
        texcoords_encoded = values["x"], values["y"], values["z"]
        size = tex.size
        x, y, z = [(v / 16383) * s - 0.5 for v, s in zip(texcoords_encoded, size)]
        ix, iy, iz = int(x + 0.5), int(y + 0.5), int(z + 0.5)
        return {
            "index": (ix, iy, iz),
            "voxel_coord": (x - ix, y - iy, z - iz),
        }


class Text(WorldObject):
    """A text.

    See :class:``pygfx.TextGeometry`` for details.

    Parameters
    ----------
    geometry : TextGeometry
        The data defining the glyphs that make up the text.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).
    name : str
        The name of the text object for inspection and debugging.

    """

    uniform_type = dict(
        WorldObject.uniform_type,
        rot_scale_transform="4x4xf4",
    )

    def __init__(
        self,
        geometry=None,
        material=None,
        *,
        visible=True,
        render_order=0,
        render_mask="auto",
        name="",
    ):
        super().__init__(
            geometry,
            material,
            visible=visible,
            render_order=render_order,
            render_mask=render_mask,
            name=name,
        )

        # calling super from callback is possible, but slow so we register it as a second callback instead
        self.world.on_update(super()._update_uniform_buffers)

    @callback
    def _update_uniform_buffers(self, transform: AffineBase):
        # super()._update_uniform_buffers(transform)
        # When rendering in screen space, the world transform is used
        # to establish the point in the scene where the text is placed.
        # The only part of the local transform that is used is the
        # position. Therefore, we also keep a transform containing the
        # local rotation and scale, so that these can be applied to the
        # text in screen coordinates.
        matrix = la.mat_compose((0, 0, 0), self.local.rotation, self.local.scale)
        self.uniform_buffer.data["rot_scale_transform"] = matrix.T
