from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.trackable import Trackable
from ..utils import array_from_shadertype, ReadOnlyDict
from ..resources import Buffer
from ..utils.enums import AlphaMethod, AlphaMode


if TYPE_CHECKING:
    from typing import Optional, ClassVar, TypeAlias, Literal, Sequence

    ABCDTuple: TypeAlias = tuple[float, float, float, float]


TEST_COMPARE_VALUES = "<", "<=", "==", "!=", ">=", ">"  # for alpha_test and depth_test


ALPHA_MODES = {
    "solid": {  # TODO: maybe the default solid should pre-multiply, so you at least see when it's wrong.
        "method": "opaque",
        "premultiply_alpha": False,
    },
    "solid_premultiply": {
        "method": "opaque",
        "premultiply_alpha": True,
    },
    "dither": {
        "method": "stochastic",
        "pattern": "blue_noise",
    },
    "bayer4": {
        "method": "stochastic",
        "pattern": "bayer4",
    },
    "blend": {
        "method": "composite",
        "color_op": "add",
        "alpha_op": "add",
        "color_constant": (0, 0, 0),
        "alpha_constant": 0,
        "color_src": "src-alpha",
        "color_dst": "one-minus-src-alpha",
        "alpha_src": "one",
        "alpha_dst": "one-minus-src-alpha",
    },
    "add": {
        "method": "composite",
        "color_op": "add",
        "alpha_op": "add",
        "color_constant": (0, 0, 0),
        "alpha_constant": 0,
        "color_src": "src-alpha",
        "color_dst": "one",
        "alpha_src": "src-alpha",
        "alpha_dst": "one",
    },
    "subtract": {
        "method": "composite",
        "color_op": "add",
        "alpha_op": "add",
        "color_constant": (0, 0, 0),
        "alpha_constant": 0,
        "color_src": "zero",
        "color_dst": "one-minus-src",
        "alpha_src": "zero",
        "alpha_dst": "one",
    },
    "multiply": {
        "method": "composite",
        "color_op": "add",
        "alpha_op": "add",
        "color_constant": (0, 0, 0),
        "alpha_constant": 0,
        "color_src": "zero",
        "color_dst": "src",
        "alpha_src": "zero",
        "alpha_dst": "src",
    },
    "weighted_blend": {
        "method": "weighted",
        "weight": "alpha",
        "alpha": "alpha",
    },
    "weighted_depth": {
        "method": "weighted",
        "weight": "alpha",
        "alpha": "alpha",
    },  # TODO: weighted blend
    "weighted_solid": {
        "method": "weighted",
        "weight": "alpha",
        "alpha": "1.0",
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
    alpha_mode : str
        How the alpha value of an object is used to to combine the resulting color
        with the target color texture.
    alpha_config : dict
        An advanced way to fully control the alpha behaviour. When both ``alpha_mode``
        and ``alpha_config`` are given, the latter is used.
    depth_test :  bool
        Whether the object takes the depth buffer into account (and how).
        Default True.
    depth_compare : str
        How to compare depth values ("<", "<=", "==", "!=", ">=", ">"). Default
        "<".
    depth_write : bool | None
        Whether the object writes to the depth buffer. With None (default) the
        value resolves to ``not material.transparency_pass``.
    alpha_test : float
        The alpha test value for this material. Default 0.0, meaning no alpha
        test is performed.
    alpha_compare : str
        How to compare alpha values ("<", "<=", "==", "!=", ">=", ">"). Default
        "<".
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
        alpha_mode: str = "solid",
        alpha_config: Optional[dict] = None,
        depth_test: bool = True,
        depth_compare: str = "<",
        depth_write: Literal[None, False, True] = None,
        pick_write: bool = False,
        alpha_test: float = 0.0,
        alpha_compare: str = "<",
    ):
        super().__init__()

        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )

        self.opacity = opacity
        self.clipping_planes = clipping_planes
        self.clipping_mode = clipping_mode
        if alpha_config is None:
            self.alpha_mode = alpha_mode
        else:
            self.alpha_config = alpha_config
        self.depth_test = depth_test
        self.depth_compare = depth_compare
        self.depth_write = depth_write
        self.pick_write = pick_write
        self.alpha_test = alpha_test
        self.alpha_compare = alpha_compare

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
        """The opacity (a.k.a. alpha value) applied to this material (0..1).

        If the material's color has an alpha smaller than 1, this alpha is multiplied with the opacity.

        Setting this value to ``<1`` will set the auto/implicit values for ``transparent`` to True,
        and the for ``depth_write`` to False.

        """
        return float(self.uniform_buffer.data["opacity"])

    @opacity.setter
    def opacity(self, value: float) -> None:
        value = min(max(float(value), 0), 1)
        self.uniform_buffer.data["opacity"] = value
        self.uniform_buffer.update_full()

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
    def alpha_mode(self) -> AlphaMode:
        """Defines how the the resulting colors are combined with the target color texture.

        There are many ways to combine the object's color with the colors of
        earlier rendered objects. The ``material.alpha_mode`` is the recommended
        way to set the ``alpha_config`` that serves the majority of use-cases.
        If more control is needed, see ``alpha_config`` to set a custom alpha
        mode. There are a range of modes to choose from, divided over four
        possible methods:

        Modes for method "opaque" (overwrites the value in the output texture):

        * "solid": alpha is ignored.
        * "solid_premultiply": the alpha is multipled with the color (making it darker).

        Modes for method "stochastic" (alpha represents the chance of a fragment being visible):

        * "dither": stochastic transparency with blue noise.
        * "bayer4": stochastic transparency with a 4x4 Bayer pattern.

        Modes for method "composite" (per-fragment blending of the object's color and the color in the output texture):

        * "blend": use classic alpha blending using the over-operator.
        * "add": use additive blending that adds the fragment color, multiplied by alpha.
        * "subtract": use subtractuve blending that removes the fragment color.
        * "multiply": use multiplicative blending that multiplies the fragment color.

        Modes for method "weighted" (order independent blending):

        * "weighted_blend": weighted blended order independent transparency.
        * "weighted_depth": weighted blended order independent transparency, weighted by depth.
        * "weighted_solid": fragments are combined based on alpha, but the final alpha is always 1. Great for e.g. image stitching.

        Note that the value of ``material.alpha_mode`` can be "custom" in case
        ``alpa_config`` is set to a configuration not covered by a prefix.
        """
        return self._store.alpha_config["mode"]

    @alpha_mode.setter
    def alpha_mode(self, alpha_mode: AlphaMode):
        if isinstance(alpha_mode, dict):
            raise TypeError(
                "material.alpha_mode must be a str, not a dict, maybe you meant material.alpha_config?"
            )
        if not isinstance(alpha_mode, str):
            raise TypeError(f"material.alpha_mode must be a str, not {alpha_mode!r}")
        alpha_mode = alpha_mode.lower()
        if alpha_mode in AlphaMethod:
            m = {
                "opaque": "solid",
                "stochastic": "dither",
                "composie": "blemd",
                "weighted": "weighted_blend",
            }
            suggestion = m.get(alpha_mode, "solid")
            raise ValueError(
                f"material.alpha_mode is set to {alpha_mode!r} which is an alpha-method. Maybe you meant {suggestion!r}?"
            )
        if alpha_mode not in AlphaMode:
            raise ValueError(
                f"material.alpha_mode must be one of {AlphaMode}, not {alpha_mode!r}"
            )
        if alpha_mode == "custom":
            raise ValueError("Cannot set material.alpha_mode with 'custom'")
        self.alpha_config = ALPHA_MODES[alpha_mode]

    @property
    def alpha_config(self) -> dict:
        """Dict that defines how the the resulting colors are combined with the target color texture.

        The ``alpha_config`` property is repesented as a dictionary that fully
        describes how the object is combined with other objects rendered at the
        same time. See ``material.alpha_mode`` for convenient presets.

        All possible alpha configurations are grouped in four methods:

        * "opaque": colors simply overwrite the texture, no transparency.
        * "stochastic": stochastic transparency, alpha represents the chance of a fragment being visible.
        * "composite": colors are blended with the buffer per-fragment.
        * "weighted": weighted blended order independent transparency, and variants thereof.

        The ``alpha_config`` dict has at least the following fields:

        * "method": select the alpha-method (mandatory when setting).
        * "mode": the corresponding mode or 'custom' (optional when setting).
        * "transparency_pass": whether the object is in the transparency-pass or the opaque-pass.
          Can be used when setting, but this override is meant for highly specific use-cases.

        For each method, different options are available, which are represented as
        items in the ``alpha_config`` dict.

        Options for method 'opaque':

        * "premultiply": whether to pre-multiply the color with the alpha values.

        Options for method 'stochastic':

        * "pattern": can be 'blue_noise' for blue noise (default), 'white_noise' for white
          noise, and 'bayer4' for a Bayer pattern. The Bayer option mixes objects
          but not elements within objects.

        Options for method 'composite':

        * "color_op": the blend operation/equation, any value from ``wgpu.BlendOperation``. Default "add".
        * "color_src": source factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * "color_dst": destination factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * "color_constant": represents the constant value of the constant blend color. Default black.
        * "alpha_op": as ``color_op`` but for alpha.
        * "alpha_src": as ``color_dst`` but for alpha.
        * "alpha_dst": as ``color_src`` but for alpha.
        * "alpha_constant": as ``color_constant`` but for alpha (default 0).

        Options for method 'weighted':

        * ``weight``: the weight factor as wgsl code. Default "alpha", which means use the color's alpha value.
        * ``alpha``: the used alpha value. Default "alpha", which means use as-is. Can e.g. be set to 1.0
          so that the alpha channel can be used as the weight factor, while the object is otherwise opaque.

        """
        return self._store.alpha_config.copy()

    @alpha_config.setter
    def alpha_config(self, alpha_config: dict) -> None:
        if not isinstance(alpha_config, dict):
            raise TypeError(
                f"material.alpha_config expects a dict, not {alpha_config!r}"
            )

        # Get the alpha method
        method = alpha_config["method"]
        if method not in AlphaMethod:
            raise ValueError(
                f"Invalid 'method' in material.alpha_config: {method!r}, expected one of {AlphaMethod}"
            )

        # Get transparency pass
        transparency_pass = method in {"composite", "weighted"}
        transparency_pass = alpha_config.get("transparency_pass", transparency_pass)

        # Init with a preset?
        mode = alpha_config.get("mode", None)
        if mode and mode != "custom":
            if mode not in ALPHA_MODES:
                raise ValueError(f"Invalid mode in material.alpha_config: {mode!r}")
            d = ALPHA_MODES[mode]
            if d["method"] != method:
                raise ValueError(f"Invalid mode {mode!r} for method {method!r}.")
            alpha_config = {**d, **alpha_config}

        if method == "opaque":
            keys = ["premultiply_alpha"]
            defaults = {}
        elif method == "stochastic":
            keys = ["pattern"]
            defaults = {"pattern": "blue_noise"}
        elif method == "composite":
            keys = [
                "color_op",
                "color_src",
                "color_dst",
                "color_constant",
                "alpha_op",
                "alpha_src",
                "alpha_dst",
                "alpha_constant",
            ]
            defaults = {
                "color_op": "add",
                "alpha_op": "add",
                "color_constant": (0, 0, 0),
                "alpha_constant": 0,
            }
        elif method == "weighted":
            keys = ["weight", "alpha"]
            defaults = {"weight": "alpha", "alpha": "alpha"}
        else:
            assert False, "if-tree is missing a case"  # noqa: B011

        # Get options. This has the same form as the values in ALPHA_MODES, so they can be compared
        options = self._get_alpha_config_options(method, keys, defaults, alpha_config)

        # Derive mode
        the_mode = "custom"
        for mode_name, mode_dict in ALPHA_MODES.items():
            if mode_dict["method"] == method:
                if mode_dict == options:
                    the_mode = mode_name
                    break

        # Save
        self._store.alpha_method = method
        self._store.alpha_config = ReadOnlyDict(
            {
                "method": method,
                "mode": the_mode,
                "transparency_pass": transparency_pass,
                **options,
            }
        )

    @property
    def transparency_pass(self):
        """Whether this object is rendered in the transparency pass.

        This is a convenient alias for ``.alpha_config['transparency_pass']``.

        This is a readonly property. To override the default derived value,
        use the 'transparency_pass' key in the alpha_config.
        """
        return self._store.alpha_config["transparency_pass"]

    def _get_alpha_config_options(
        self, method: str, keys: list, default_dict: dict, given_dict: dict
    ):
        err_preamble = f"material.alpha_config for method {method!r}"
        assert isinstance(given_dict, dict)
        source_dict = given_dict.copy()
        result_dict = {"method": method, **default_dict}
        for key in ["method", "mode", "transparency_pass"]:
            source_dict.pop(key, None)
        try:
            for key in keys:
                if key in source_dict:
                    result_dict[key] = source_dict.pop(key)
        except KeyError as err:
            raise KeyError(f"{err_preamble} is missing field {err.args[0]!r}") from None
        if source_dict:
            raise ValueError(f"{err_preamble} contains invalid fields: {source_dict!r}")
        return result_dict

    @property
    def depth_test(self) -> bool:
        """Whether the object takes the depth buffer into account.

        When set to True, the fragment's depth is tested using ``depth_compare`` against the depth buffer.
        """
        return self._store.depth_test

    @depth_test.setter
    def depth_test(self, value: bool) -> None:
        self._store.depth_test = bool(value)

    @property
    def depth_compare(self):
        """The way to compare the depth with the value in the buffer.

        Possible values are "<", "<=", "==", "!=", ">=", ">". Default "<".
        Note that this only applies if ``depth_test`` is set to True.
        """
        return self._store.depth_compare

    @depth_compare.setter
    def depth_compare(self, value) -> None:
        if not (isinstance(value, str) and value in TEST_COMPARE_VALUES):
            raise TypeError(
                "Material.depth_compare must be a str in {TEST_COMPARE_VALUES!r}, not {value!r}"
            )
        self._store.depth_compare = value

    @property
    def depth_write(self) -> bool:
        """Whether this material writes to the depth buffer, preventing other objects being drawn behind it.

        Can be set to:

        * True: yes, write depth.
        * False: no, don't write depth.
        * None: auto-determine (default): ``not transparency_pass``.

        The auto-option provides good default behaviour for common use-case, but
        if you know what you're doing you should probably just set this value to
        True or False.
        """
        depth_write = self._store.depth_write
        if depth_write is None:
            depth_write = not self.transparency_pass
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
    def alpha_test(self) -> float:
        """The alpha test value for this material.

        When ``alpha_test`` is set to a value > 0, the fragment is discarded if ``alpha < alpha_test``.
        This is useful for e.g. grass or foliage textures, where the texture has a lot of transparent
        areas. Also see ``alpha_compare``.
        """
        return self.uniform_buffer.data["alpha_test"]

    @alpha_test.setter
    def alpha_test(self, value: float) -> None:
        value = min(max(float(value), 0), 1)
        self.uniform_buffer.data["alpha_test"] = value
        self.uniform_buffer.update_full()
        # Store whether the alpha test is active, so we can invalidate shaders
        self._store.use_alpha_test = value != 0

    @property
    def alpha_compare(self) -> str:
        """The way to compare the alpha value.

        Possible values are "<", "<=", "==", "!=", ">=", ">". Default "<".
        Note that this only applies if the alpha test is performed (i.e. ``alpha_test`` is nonzero).
        """
        return self._store.alpha_compare

    @alpha_compare.setter
    def alpha_compare(self, value: str) -> None:
        if not (isinstance(value, str) and value in TEST_COMPARE_VALUES):
            raise TypeError(
                "Material.alpha_compare must be a str in {TEST_COMPARE_VALUES!r}, not {value!r}"
            )
        self._store.alpha_compare = value

    @property
    def _gfx_use_alpha_test(self) -> bool:
        """For internal use; if the alpha test is used, the shader must discard, which we don't want to do unless needed."""
        return self._store.use_alpha_test

    @property
    def pick_write(self) -> bool:
        """Whether this material is picked by the pointer."""
        return self._store.pick_write

    @pick_write.setter
    def pick_write(self, value: int) -> None:
        if not isinstance(value, (bool, int)):
            raise TypeError("Material.pick_write must be bool.")
        self._store.pick_write = bool(value)
