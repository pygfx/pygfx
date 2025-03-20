from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.trackable import Trackable
from ..utils import array_from_shadertype
from ..resources import Buffer


if TYPE_CHECKING:
    from typing import Union, ClassVar, TypeAlias, Literal, Sequence

    ABCDTuple: TypeAlias = tuple[float, float, float, float]


class Material(Trackable):
    """Material base class.

    Parameters
    ----------
    opacity : float
        The opacity (a.k.a. alpha value) applied to this material, expressed as
        a value between 0 and 1. If the material contains any non-opaque
        fragments, their alphas are simply scaled by this value.
    clipping_planes : tuple
        A tuple of planes (abcd tuples) in world space. Points in space whose
        signed distance to the plane is negative are clipped (not rendered).
        Applies to the object to which this material is attached.
    clipping_mode : str
        Set the behavior for multiple clipping planes: "any" or "all". If this
        is "any" (the default) a fragment is discarded if it is clipped by any
        clipping plane. If this is "all", a fragment is discarded only if it is
        clipped by *all* of the clipping planes.
    transparent : bool | None
        Whether the object is (semi) transparent.
        Default (None) tries to derive this from the shader.
    blending : str : dict
        The way to blend semi-transparent fragments  (alpha < 1) for this material.
    depth_test : bool
        Whether the object takes the depth buffer into account.
        Default True. If False, the object is like a ghost: not testing
        against the depth buffer and also not writing to it.
    depth_write :  bool | None
        Whether the object writes to the depth buffer. Default (None' is
        based on transparency.
    """

    # Note that in the material classes we define what properties are stored as
    # a uniform, and which are stored on `._store` (and will likely be used as
    # shader templating variables). This can be seen as an abstraction leak, but
    # abstracing it away is inpractical and complex. So we embrace wgpu as the
    # primary rendering backend, and other backends should deal with it.
    # See https://github.com/pygfx/pygfx/issues/272

    uniform_type: ClassVar[dict[str, str]] = dict(
        opacity="f4",
        clipping_planes="0*4xf4",  # array<vec4<f32>,3>
    )

    def __init__(
        self,
        *,
        opacity: float = 1,
        clipping_planes: Sequence[ABCDTuple] = (),
        clipping_mode: Literal["ANY", "ALL"] = "ANY",
        transparent: Union[bool, None] = None,
        blending: str = "normal",
        depth_test: bool = True,
        depth_write: Union[bool, None] = None,
        pick_write: bool = False,
    ):
        super().__init__()

        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )

        self.opacity = opacity
        self.clipping_planes = clipping_planes
        self.clipping_mode = clipping_mode
        self.transparent = transparent
        self.blending = blending
        self.depth_test = depth_test
        self.depth_write = depth_write
        self.pick_write = pick_write

    def _set_size_of_uniform_array(self, key: str, new_length: int) -> None:
        """Resize the given array field in the uniform struct if the
        current length does not match the given length. This adjusts
        the uniform type, creates a new buffer from that, and copies
        all data except that of the given field.

        Resetting the uniform buffer will bump the rev and thus trigger
        a pipeline rebuild for all objects that this material is
        attached to.
        """
        current_length = self.uniform_buffer.data[key].shape[0]
        if new_length == current_length:
            return  # early exit

        format = self.uniform_type[key]
        assert "*" in format, f"uniform field {key} '{format}' is not an array"
        n, _, subtype = format.partition("*")
        assert int(n) >= 0

        # Adjust type definition (note that this is originally a class attr)
        self.uniform_type[key] = f"{new_length}*{subtype}"
        # Recreate buffer
        data = self.uniform_buffer.data
        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )
        # Copy data
        for k in data.dtype.names:
            if k != key:
                self.uniform_buffer.data[k] = data[k]

    def _wgpu_get_pick_info(self, pick_value) -> dict:
        """Given a 4 element tuple, sampled from the pick texture,
        return info about what was picked in the object. The first value
        refers to the object id. The use of the remaining values differs
        per material.
        """
        # Note that this is a private friend-method of the renderer.
        return {}

    @property
    def uniform_buffer(self) -> Buffer:
        """The uniform buffer object for this material.

        Properties that are represented in the buffer can be updated cheaply
        (i.e. without requiring shader compilation).
        """
        return self._store.uniform_buffer

    @property
    def opacity(self) -> float:
        """The opacity (a.k.a. alpha value) applied to this material, expressed
        as a value between 0 and 1. If the material contains any
        non-opaque fragments, their alphas are simply scaled by this value.
        """
        return float(self.uniform_buffer.data["opacity"])

    @opacity.setter
    def opacity(self, value: float) -> None:
        value = min(max(float(value), 0), 1)
        self.uniform_buffer.data["opacity"] = value
        self.uniform_buffer.update_full()
        self._store.is_transparent = value < 1

    # todo: rename to _opacity_is_transparent
    @property
    def is_transparent(self) -> bool:
        """Whether this material is (semi) transparent because of its opacity value.
        If False the object can still appear transparent because of its color.
        """
        return self._store.is_transparent

    @property
    def clipping_planes(self) -> Sequence[ABCDTuple]:
        """A tuple of planes (abcd tuples) in world space. Points in
        space whose signed distance to the plane is negative are clipped
        (not rendered). Applies to the object to which this material is attached.
        """
        return [
            tuple(float(f) for f in plane.flat)  # type: ignore
            for plane in self.uniform_buffer.data["clipping_planes"]
        ]

    @clipping_planes.setter
    def clipping_planes(self, planes: Sequence[ABCDTuple]):
        if not isinstance(planes, (tuple, list)):
            raise TypeError("Clipping planes must be a list.")
        planes2 = []
        for plane in planes:
            if isinstance(plane, (tuple, list)) and len(plane) == 4:
                planes2.append(plane)
            # maybe someday elif isinstance(plane, linalg.Plane):
            else:
                # Error
                if isinstance(plane, (int, float)) and len(planes) == 4:
                    raise TypeError(
                        "Looks like you passed one clipping plane instead of a list."
                    )
                else:
                    raise TypeError(
                        f"Each clipping plane must be an abcd tuple, not {plane}"
                    )

        # Apply
        # Note that we can create a null-plane using (0, 0, 0, 0)
        self._set_size_of_uniform_array("clipping_planes", len(planes2))
        self._store.clipping_plane_count = len(planes2)
        for i in range(len(planes2)):
            self.uniform_buffer.data["clipping_planes"][i] = planes2[i]
        self.uniform_buffer.update_full()

    @property
    def clipping_plane_count(self) -> int:
        """The number of clipping planes (readonly)."""
        return self._store.clipping_plane_count

    @property
    def clipping_mode(self) -> Literal["ANY", "ALL"]:
        """Set the behavior for multiple clipping planes: "ANY" or "ALL".
        If this is "ANY" (the default) a fragment is discarded
        if it is clipped by any clipping plane. If this is "ALL", a
        fragment is discarded only if it is clipped by *all* of the
        clipping planes.
        """
        return self._clipping_mode

    @clipping_mode.setter
    def clipping_mode(self, value: Literal["ANY", "ALL"]) -> None:
        mode = value.upper()
        if mode in ("ANY", "ALL"):
            self._clipping_mode = mode
        else:
            raise ValueError(f"Unexpected clipping_mode: {value}")

    @property
    def transparent(self) -> Union[bool, None]:
        """Whether this material is (semi) transparent.

        * True: consider the object as transparent.
        * False: consider the object as opaque.
        * None: auto-determine based on ``.opacity``, ``.color`` and more.
        """
        return self._store.transparent

    @transparent.setter
    def transparent(self, value: Union[bool, None]) -> None:
        if value is None:
            self._store.transparent = None
        elif isinstance(value, bool):
            self._store.transparent = bool(value)
        else:
            raise TypeError("material.transparent must be bool or None.")

    @property
    def blending(self):
        """The way to blend semi-transparent fragments (alpha < 1) for this material.

        * "no": No blending, render as opaque.
        * "normal": Alpha blending using the over operator (the default).
        * "add": Add the fragment color, multiplied by alpha.
        * "subtract": Subtract the fragment color, multiplied by alpha.
        * "dither": use stochastic blending. All fragments are opaque, and the chance
          of a fragment being discared (invisible) is one minus alpha.
        * "weighted": use weighted blending, where the order of objects does not matter for the
          end-result.
        * A dict: Custom blend function (equation, src, dst). TODO

        """
        return self._store.blending

    @blending.setter
    def blending(self, blending):
        if blending is None:
            blending = "normal"
        valids = ["no", "normal", "add", "subtract", "dither", "weighted"]
        if isinstance(blending, str):
            if blending not in valids:
                raise ValueError(
                    f"Blending is {blending!r} but expected one of {valids!r}"
                )
        elif isinstance(blending, dict):
            raise NotImplementedError()
        else:
            raise TypeError("Material.blending must be None, str or dict.")
        self._store.blending = blending

    @property
    def depth_test(self) -> bool:
        """Whether the object takes the depth buffer into account."""
        return self._store.depth_test

    @depth_test.setter
    def depth_test(self, value: bool) -> None:
        # Explicit test that this is a bool. We *could* maybe later allow e.g. 'greater'.
        if not isinstance(value, (bool, int)):
            raise TypeError("Material.depth_test must be bool.")
        self._store.depth_test = bool(value)

    @property
    def depth_write(self) -> Union[bool, None]:
        """Whether this material writes to the depth buffer, preventing futher writing behind it.

        * True: yes, write depth.
        * False: no, don't write depth.
        * None: auto-determine based on ``blending`` and (detected) transparency.
        """
        return self._store.depth_write

    @depth_write.setter
    def depth_write(self, value: Union[bool, None]) -> None:
        if value is None:
            self._store.depth_write = None
        elif isinstance(value, bool):
            self._store.depth_write = bool(value)
        else:
            raise TypeError("material.depth_write must be bool or None.")

    @property
    def pick_write(self) -> bool:
        """Whether this material is picked by the pointer."""
        return self._store.pick_write

    @pick_write.setter
    def pick_write(self, value: int) -> None:
        if not isinstance(value, (bool, int)):
            raise TypeError("Material.pick_write must be bool.")
        self._store.pick_write = bool(value)
