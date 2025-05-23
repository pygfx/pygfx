from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.trackable import Trackable
from ..utils import array_from_shadertype, ReadOnlyDict
from ..resources import Buffer


if TYPE_CHECKING:
    from typing import Union, ClassVar, TypeAlias, Literal, Sequence

    ABCDTuple: TypeAlias = tuple[float, float, float, float]


# The blending presets
# note: we assume alpha is not pre-multiplied
preset_blending_dicts = {
    "no": {
        "mode": "classic",
        "color_src": "one",
        "color_dst": "zero",
        "alpha_src": "one",
        "alpha_dst": "zero",
    },
    "normal": {
        "mode": "classic",
        "color_src": "src-alpha",
        "color_dst": "one-minus-src-alpha",
        "alpha_src": "one",
        "alpha_dst": "one-minus-src-alpha",
    },
    "additive": {
        "mode": "classic",
        "color_src": "src-alpha",
        "color_dst": "one",
        "alpha_src": "src-alpha",
        "alpha_dst": "one",
    },
    "subtractive": {
        "mode": "classic",
        "color_src": "zero",
        "color_dst": "one-minus-src",
        "alpha_src": "zero",
        "alpha_dst": "one",
    },
    "multiply": {
        "mode": "classic",
        "color_src": "zero",
        "color_dst": "src",
        "alpha_src": "zero",
        "alpha_dst": "src",
    },
    "dither": {
        "mode": "dither",
    },
    "weighted": {
        "mode": "weighted",
    },
}


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
    blending : str | dict
        The way to blend semi-transparent fragments  (alpha < 1) for this material.
    depth_test :  bool
        Whether the object takes the depth buffer into account.
        Default True. If False, the object is like a ghost: not testing
        against the depth buffer and also not writing to it.
    depth_write : bool | None
        Whether the object writes to the depth buffer. With None (default) this
        is determined automatically.
    alpha_test : float
        The alpha test value for this material. Default 0.0, meaning no alpha
        test is performed.
    """

    # Note that in the material classes we define what properties are stored as
    # a uniform, and which are stored on `._store` (and will likely be used as
    # shader templating variables). This can be seen as an abstraction leak, but
    # abstracing it away is inpractical and complex. So we embrace wgpu as the
    # primary rendering backend, and other backends should deal with it.
    # See https://github.com/pygfx/pygfx/issues/272

    uniform_type: ClassVar[dict[str, str]] = dict(
        opacity="f4",
        alpha_test="f4",
        clipping_planes="0*4xf4",  # array<vec4<f32>,3>
    )

    def __init__(
        self,
        *,
        opacity: float = 1,
        clipping_planes: Sequence[ABCDTuple] = (),
        clipping_mode: Literal["ANY", "ALL"] = "ANY",
        transparent: Literal[None, False, True] = None,
        blending: Union[str, dict] = "normal",
        depth_test: bool = True,
        depth_write: Literal[None, False, True] = None,
        pick_write: bool = False,
        alpha_test: float = 0.0,
    ):
        super().__init__()

        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )

        self._user_transparent = None

        self.opacity = opacity
        self.clipping_planes = clipping_planes
        self.clipping_mode = clipping_mode
        self.transparent = transparent
        self.blending = blending
        self.depth_test = depth_test
        self.depth_write = depth_write
        self.pick_write = pick_write
        self.alpha_test = alpha_test

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
        self._resolve_transparent()

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
    def transparent(self) -> Literal[None, False, True]:
        """Defines whether this material is transparent or opaque.

        If set to None, the transparency is autodetermined
        based on ``.opacity`` and possibly other material properties.

        The final transparency value is one of:

        * False: the object is (considered) fully opaque. The renderer draws
          these first, and sorts front-to-back to avoid overdrawing.
        * True: the object is (considered) fully transparent. The renderer draws
          these after opaque objects, and sorts back-to-front to increase the chance of correct blending.
        * None: the object is considered to (possibly) have both opaque and transparent
          fragments. The renderer draws these in between opaque and transparent
          passes, back-to-front.
        """
        return self._store.transparent

    @transparent.setter
    def transparent(self, value: Literal[None, False, True]):
        if value is None:
            self._user_transparent = None
        elif isinstance(value, bool):
            self._user_transparent = bool(value)
        else:
            raise TypeError("material.transparent must be bool or None.")
        self._resolve_transparent()

    @property
    def transparent_is_set(self):
        """Whether the ``transparent`` property is set. Otherwise it's auto-determined."""
        return self._user_transparent is not None

    def _resolve_transparent(self):
        # This method must be called from the appropriate places
        transparent = self._user_transparent
        if transparent is None:
            looks_transparent = self._looks_transparent()
            if looks_transparent is not None:
                transparent = bool(looks_transparent)
        self._store.transparent = transparent

    def _looks_transparent(self):
        # This could be overloaded in subclasses.
        transparent = None
        if self.opacity < 1:
            transparent = True
        return transparent

    @property
    def blending(self):
        """The way to blend semi-transparent fragments (alpha < 1) for this material.

        The blending can be set using one of the following preset names:

        * "no": no blending, render as opaque.
        * "normal": use classic alpha blending using the 'over' operator (the default).
        * "additive": use additive blending that adds the fragment color, multiplied by alpha.
        * "subtractive": use subtractuve blending that removes the fragment color.
        * "multiply": use multiplicative blending that multiplies the fragment color.
        * "dither": use stochastic transparency blending. All fragments are opaque, and the chance
          of a fragment being discared (invisible) is one minus alpha.
        * "weighted": use weighted blending, where the order of objects does not matter for the end-result.

        The blending property returns a dict. It can also be set as a dict. Such a dict has the following fields:

        * "name": the preset name of this blending, or 'custom'. (This field is ignored when setting the ``blending``.)
        * "mode": the blend-mode, one of "classic", "dither", "weighted".
        * When ``mode`` is "classic", the following fields must/can be provided:
          * "color_op": the blend operation/equation, any value from ``wgpu.BlendOperation``. Default 'add'.
          * "color_src": source factor, any value of ``wgpu.BlendFactor``. Mandatory.
          * "color_dst": destination factor, any value of ``wgpu.BlendFactor``. Mandatory.
          * "color_constant": represents the constant value of the constant blend color. Default black.
          * "alpha_op": as ``color_op`` but for alpha.
          * "alpha_src": as ``color_dst`` but for alpha.
          * "alpha_dst": as ``color_src`` but for alpha.
          * "alpha_constant": as ``color_constant`` but for alpha (default 0).
        * When ``mode`` is "dither": there are (currently) no extra fields.
        * When ``mode`` is "weighted":
          * "weight": the weight factor as wgsl code. Default 'alpha', which means use the color's alpha value.
          * "alpha": the used alpha value. Default 'alpha', which means use as-is. Can e.g. be set to 1.0
            so that the alpha channel can be used as the weight factor, while the object is otherwise opaque.

        """
        return self._store.blending_dict

    @blending.setter
    def blending(self, blending):
        if blending is None:
            blending = "normal"

        if isinstance(blending, str):
            preset_keys = set(preset_blending_dicts.keys())
            if blending not in preset_keys:
                raise ValueError(
                    f"Blending is {blending!r} but expected one of {preset_keys!r}"
                )
            blending_dict = preset_blending_dicts[blending]

        elif isinstance(blending, dict):
            # we pop elements from the given dict, so we can detect invalid fields
            blending_src = blending.copy()
            blending_dict = {}
            try:
                blending_src.pop("name", None)
                blending_dict["mode"] = mode = blending_src.pop("mode")
                if mode == "classic":
                    for key in ["color_src", "color_dst", "alpha_src", "alpha_dst"]:
                        blending_dict[key] = blending_src.pop(key)
                    for key in [
                        "color_op",
                        "alpha_op",
                        "color_constant",
                        "alpha_constant",
                    ]:
                        if key in blending_src:
                            blending_dict[key] = blending_src.pop(key)
                elif mode == "dither":
                    pass
                elif mode == "weighted":
                    for key in ["weight", "alpha"]:
                        if key in blending_src:
                            blending_dict[key] = blending_src.pop(key)
                else:
                    raise ValueError(f"Unexpected blending mode {mode!r}")
            except KeyError as err:
                raise KeyError(
                    f"Blending dict is missing field {err.args[0]!r}"
                ) from None
            if blending_src:
                raise ValueError(
                    f"Blending dict contains invalid fields: {blending_src!r}"
                )

        else:
            raise TypeError("Material.blending must be None, str or dict.")

        # Prepend the preset name if the dict matches a preset
        for preset_name, preset_dict in preset_blending_dicts.items():
            if blending_dict == preset_dict:
                blending_dict = {"name": preset_name, **blending_dict}
                break
        else:
            blending_dict = {"name": "custom", **blending_dict}

        self._store.blending_dict = ReadOnlyDict(blending_dict)
        self._store.blending_mode = blending_dict["mode"]

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
    def depth_write(self) -> bool:
        """Whether this material writes to the depth buffer, preventing other objects being drawn behind it.

        Can be set to:

        * True: yes, write depth.
        * False: no, don't write depth.
        * None: auto-determine (default): yes unless ``.transparent`` is True.

        The auto-option provides good default behaviour for common use-case, but
        if you know what you're doing you should probably just set this value to
        True or False.
        """
        depth_write = self._store.depth_write
        if depth_write is None:
            if self._store.blending_mode == "dither":
                # The great thing with stochastic transparency is that everything is opaque (from a depth testing perspective)
                depth_write = True
            else:
                # Write depth when transparent is False or None
                depth_write = not bool(self._store.transparent)
        return depth_write

    @depth_write.setter
    def depth_write(self, value: Literal[None, False, True]) -> None:
        if value is None:
            self._store.depth_write = None
        elif isinstance(value, bool):
            self._store.depth_write = bool(value)
        else:
            raise TypeError("material.depth_write must be bool or None.")

    @property
    def depth_write_is_set(self):
        """Whether the ``depth_write`` property is set. Otherwise it's auto-determined."""
        return self._store.depth_write is not None

    @property
    def alpha_test(self) -> bool:
        """The alpha test value for this material.

        When ``alpha_test`` is set to a value > 0, the fragment is discarded if ``alpha < alpha_test``.
        This is useful for e.g. grass or foliage textures, where the texture has a lot of transparent
        areas. When it is set to a value < 0, the fragment is discarded if ``alpha > abs(alpha_test)``.
        """
        return self.uniform_buffer.data["alpha_test"]

    @alpha_test.setter
    def alpha_test(self, value: float) -> None:
        value = min(max(float(value), -1), 1)
        self.uniform_buffer.data["alpha_test"] = value
        self.uniform_buffer.update_full()
        # Store whether the alpha test is active, so we can invalidate shaders
        self._store.uses_alpha_test = value != 0

    @property
    def _gfx_uses_alpha_test(self) -> bool:
        """For internal use; if the alpha test is used, the shader must discard, which we don't want to do unless needed."""
        return self._store.uses_alpha_test

    @property
    def pick_write(self) -> bool:
        """Whether this material is picked by the pointer."""
        return self._store.pick_write

    @pick_write.setter
    def pick_write(self, value: int) -> None:
        if not isinstance(value, (bool, int)):
            raise TypeError("Material.pick_write must be bool.")
        self._store.pick_write = bool(value)
