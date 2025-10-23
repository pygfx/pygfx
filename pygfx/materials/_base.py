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
    "solid": {
        "method": "opaque",
        "premultiply_alpha": False,
    },
    "solid_premul": {
        "method": "opaque",
        "premultiply_alpha": True,
    },
    "blend": {
        "method": "blended",
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
        "method": "blended",
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
        "method": "blended",
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
        "method": "blended",
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
    "weighted_solid": {
        "method": "weighted",
        "weight": "alpha",
        "alpha": "1.0",
    },
    "dither": {
        "method": "stochastic",
        "pattern": "blue_noise",
        "seed": "element",
    },
    "bayer": {
        "method": "stochastic",
        "pattern": "bayer",
        "seed": "object",  # bc not enough variation to do 'element'
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
        value depends on the ``alpha_config``.
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
        alpha_mode: Optional[str] = "auto",
        alpha_config: Optional[dict] = None,
        depth_test: bool = True,
        depth_compare: str = "<",
        depth_write: Literal[None, False, True] = None,
        pick_write: bool = False,
        alpha_test: float = 0.0,
        alpha_compare: str = "<",
        render_queue: Optional[int] = None,
    ):
        super().__init__()

        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )
        self._given_render_queue = 2000  # init to avoid resolving many times
        self._store["use_alpha_test"] = None
        self.opacity = opacity
        self.clipping_planes = clipping_planes
        self.clipping_mode = clipping_mode
        if alpha_config is None:
            self.alpha_mode = alpha_mode or "auto"
        else:
            self.alpha_config = alpha_config
        self.depth_test = depth_test
        self.depth_compare = depth_compare
        self.depth_write = depth_write
        self.pick_write = pick_write
        self.alpha_test = alpha_test
        self.alpha_compare = alpha_compare
        self.render_queue = render_queue  # set last, bc it resolves from other values

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

        The ``material.alpha_mode`` is the recommended way to specify how an
        object's color is combined with the colors of earlier rendered objects.
        It provides preset configurations covering the majority of use-cases.

        The ``alpha_mode`` is a convenient way to set ``alpha_method`` and ``alpha_config`.
        If more control is needed, set ``alpha_config`` directly.

        Modes for method "opaque" (overwrites the value in the output texture):

         * "solid": alpha is ignored.
         * "solid_premul": the alpha is multipled with the color (making it darker).

        Modes for method "blended" (per-fragment blending, a.k.a. compositing):

         * "auto": classic alpha blending, with ``depth_write`` defaulting to True. See note below.
         * "blend": classic alpha blending using the over-operator. ``depth_write`` defaults to False.
         * "add": additive blending that adds the fragment color, multiplied by alpha.
         * "subtract": subtractuve blending that removes the fragment color.
         * "multiply": multiplicative blending that multiplies the fragment color.

        Modes for method "weighted" (order independent blending):

         * "weighted_blend": weighted blended order independent transparency.
         * "weighted_solid": fragments are combined based on alpha, but the final alpha is always 1. Great for e.g. image stitching.

        Modes for method "stochastic" (alpha represents the chance of a fragment being visible):

         * "dither": stochastic transparency with blue noise. This mode handles
           order-independent transparency exceptionally well, but it produces
           results that can look somewhat noisy.
         * "bayer": stochastic transparency with an 8x8 Bayer pattern.

        Note that the special mode "auto" produces reasonable results for common
        use-cases and can handle objects that produce a mix of opaque (alpha=1)
        and transparent (alpha<1) fragments, i.e. it can handle lines, points,
        and text with ``material.aa=True``. Artifacts can occur when objects are rendered out
        of order and/or when objects intersect. A different method such as
        "blend", "dither", or "weighted_blend" is then recommended.

        Note that for methods 'opaque' and 'stochastic', the ``depth_write``
        defaults to True, and for methods 'blended' and 'weighted' the
        ``depth_write`` defaults to False (except when mode is 'auto').

        Note that the value of ``material.alpha_mode`` can be "custom" in case
        ``alpa_config`` is set to a configuration not covered by a preset mode.
        """
        return self._store.alpha_mode

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
                "blended": "blend",
                "weighted": "weighted_blend",
                "stochastic": "dither",
            }
            suggestion = m.get(alpha_mode, "solid")
            raise ValueError(
                f"material.alpha_mode is set to {alpha_mode!r} which is an alpha-method. Maybe you meant {suggestion!r}?"
            )
            if alpha_mode not in AlphaMode:
                raise ValueError("Cannot set material.alpha_mode with 'custom'")

        if alpha_mode == "auto":
            d = ALPHA_MODES["blend"].copy()
            d["mode"] = "auto"
        else:
            d = ALPHA_MODES[alpha_mode]
        self.alpha_config = d

    @property
    def alpha_method(self) -> str:
        """The alpha method being used (readonly).

        The alpha method determines the main way how alpha values are used. There are four options:

        * "opaque" means the fragments overwrite value in the color buffer.
        * "blended" means the fragments are blended (composited) with the color buffer.
        * "weighted" means the fragments are blended in a weighted order-independent way.
        * "stochastic" means the fragments are opaque, and alpha represents the chance of a fragment being visible.

        The ``alpha_config`` dictionary specifies extra parameters that affect the behavour of each alpha method.
        The ``alpha_mode`` is a convenient way to set ``alpha_config`` (and ``alpha_method``) using presets

        """
        return self._store.alpha_method

    @property
    def alpha_config(self) -> dict:
        """Dict that defines how the the resulting colors are combined with the target color texture.

        The ``alpha_config`` property is repesented as a dictionary that fully
        describes how the object is combined with other objects rendered at the
        same time. See ``material.alpha_mode`` for convenient presets.

        All possible alpha configurations are grouped in four methods:

        * "opaque": colors simply overwrite the texture, no transparency.
        * "blended": colors are blended with the buffer per-fragment.
        * "weighted": weighted blended order independent transparency, and variants thereof.
        * "stochastic": stochastic transparency, alpha represents the chance of a fragment being visible.

        The ``alpha_config`` dict has at least the following fields:

        * "mode": the corresponding mode or 'custom' (optional when setting).
        * "method": select the alpha-method (mandatory when setting).

        For each method, different options are available, which are represented as
        items in the ``alpha_config`` dict.

        Options for method 'opaque':

        * "premultiply": whether to pre-multiply the color with the alpha values.

        Options for method 'blended':

        * "color_op": the blend operation/equation, any value from ``wgpu.BlendOperation``. Default "add".
        * "color_src": source factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * "color_dst": destination factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * "color_constant": represents the constant value of the constant blend color. Default black.
        * "alpha_op": as ``color_op`` but for alpha.
        * "alpha_src": as ``color_dst`` but for alpha.
        * "alpha_dst": as ``color_src`` but for alpha.
        * "alpha_constant": as ``color_constant`` but for alpha (default 0).

        Options for method 'weighted':

        * "weight"``": the weight factor as wgsl code. Default "alpha", which means use the color's alpha value.
        * "alpha": the used alpha value. Default "alpha", which means use as-is. Can e.g. be set to 1.0
          so that the alpha channel can be used as the weight factor, while the object is otherwise opaque.

        Options for method 'stochastic':

        * "pattern": can be 'blue-noise' for blue noise (default), 'white-noise' for white
          noise, and 'bayer' for a Bayer pattern.
        * "seed": can be 'screen' to have a uniform pattern for the whole screen, 'object' to
          use a per-object seed, and 'element' to have a per-element seed.
          The default is 'element' for  the noise patterns and 'object' for the bayer pattern.


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

        # Init with a preset?
        mode = alpha_config.get("mode", None)
        if mode and mode not in ("auto", "custom"):
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
            keys = ["pattern", "seed"]
            defaults = {"pattern": "blue_noise", "seed": "screen"}
        elif method == "blended":
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
        if mode == "auto" and the_mode == "blend":
            the_mode = "auto"

        # Save
        self._store.alpha_mode = the_mode
        self._store.alpha_method = method
        self._store.alpha_config = ReadOnlyDict(
            {
                "mode": the_mode,
                "method": method,
                **options,
            }
        )
        self._derive_render_queue()

    @property
    def render_queue(self) -> int:
        """An integer that represents the group that the renderer uses to sort objects.

        The property is intended for advanced usage; by default it is determined automatically
        based on ``alpha_mode``, ``alpha_method`` and ``alpha_test``.

        The ``render_queue`` is a number between 1 and 4999. The builtin values are:

        * 1000: background.
        * 2000: opaque non-blending objects.
        * 2400: opaque objects with a discard based on alpha (i.e. using ``alpha_test`` or "stochastic" alpha-method).
        * 2600: objects with alpha-mode 'auto'.
        * 3000: transparent objects.
        * 4000: overlay.

        Objects with ``render_queue`` between 1501 and 2500 are sorted front-to-back. Otherwise
        objects are sorted back-to-front.

        It can be tempting to use ``material.render_queue`` to control the render order of individual objects,
        but for that purpose ``ob.render_order`` is more appropriate.
        """
        return self._render_queue

    @render_queue.setter
    def render_queue(self, render_queue: int | None):
        render_queue = int(render_queue or 0)
        if render_queue < 0 or render_queue > 4999:
            if render_queue == -1:
                render_queue = 0
            else:
                raise ValueError(
                    "Material.render_queue must be an int between 1 and 4999, or 0 for 'default'."
                )
        self._given_render_queue = render_queue
        self._derive_render_queue()

    @property
    def render_queue_is_set(self):
        """Whether the ``render_queue`` property is set. Otherwise it's auto-determined."""
        return bool(self._given_render_queue)

    def _derive_render_queue(self):
        if self._given_render_queue:
            render_queue = self._given_render_queue
        elif self._store.alpha_mode == "auto":
            # An "opaque" auto-object, but it may have semi-transparent fragments,
            # because of aa, or maps with transparent regions. So back-to-front.
            render_queue = 2600
        else:
            alpha_method = self._store.alpha_method
            if alpha_method == "opaque":
                if not self._store.use_alpha_test:
                    render_queue = 2000
                else:
                    render_queue = 2400
            elif alpha_method == "stochastic":
                render_queue = 2400
            else:  # alpha_method in ["blended", "weighted"]
                render_queue = 3000
        self._render_queue = render_queue

    def _get_alpha_config_options(
        self, method: str, keys: list, default_dict: dict, given_dict: dict
    ):
        err_preamble = f"material.alpha_config for method {method!r}"
        assert isinstance(given_dict, dict)
        source_dict = given_dict.copy()
        result_dict = {"method": method, **default_dict}
        for key in ["method", "mode"]:
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
        * None: auto-determine based on ``alpha_coalpha_mode`` and ``alpha_method`` (default).

        The auto-option provides good default behaviour for common use-case, but
        if you know what you're doing you should probably just set this value to
        True or False.
        """
        depth_write = self._store.depth_write
        if depth_write is None:
            depth_write = (
                self._store.alpha_mode == "auto"
                or self._store.alpha_method in ("opaque", "stochastic")
            )
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
        use_alpha_test = value != 0
        if use_alpha_test != self._store.use_alpha_test:
            self._store.use_alpha_test = use_alpha_test
            self._derive_render_queue()

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
