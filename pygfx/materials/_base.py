from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.trackable import Trackable
from ..utils import array_from_shadertype, ReadOnlyDict
from ..resources import Buffer
from ..utils.enums import AlphaMode


if TYPE_CHECKING:
    from typing import Union, ClassVar, TypeAlias, Literal, Sequence

    ABCDTuple: TypeAlias = tuple[float, float, float, float]


TEST_COMPARE_VALUES = "<", "<=", "==", "!=", ">=", ">"  # for alpha_test and depth_test


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
        TODO if we want users to usually set this, can we think of a one-word param?
        How the alpha value of an object is used to to combine the resulting
        color with the target color texture.
    depth_test :  bool
        Whether the object takes the depth buffer into account (and how).
        Default True.
    depth_compare : str
        How to compare depth values ("<", "<=", "==", "!=", ">=", ">"). Default
        "<".
    depth_write : bool | None
        Whether the object writes to the depth buffer. With None (default) the value
        resolves to ``alpha_mode in ('opaque', 'dither')``.
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
        alpha_mode: str = "opaque",
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

        self.blend_mode = None
        self.dither_mode = None
        self.weighted_mode = None
        self.alpha_mode = alpha_mode

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
        """How the alpha value of an object is used to to combine the resulting color with the target color texture.

        The ``alpha_mode`` has one of the following base values.

        * "opaque": Fragments overwrite the value in the color texture. The
          object is considered opaque (i.e. not transparent) regardless of the
          alpha value. The fragment color is multiplied with alpha, i.e. making it darker if alpha is ``<1``.
          The ``depth_write`` defaults to True.
        * "blend": Per-fragment blending of the object's color and the color in
          the color texture (a.k.a. alpha compositing). The result depends on
          the order in which objects are drawn. The renderer sorts objects from
          back to front, but does not account for (self)intersections. If this
          is a problem, use "dither" or "weighted" for (some of) your
          transparent objects. The ``depth_write`` defaults to False.
        * "dither": Stochastic transparency. The chance of a fragment being
          written (not discarded) is equal to alpha, and all fragments are
          opaque. This mode is invariant to the order in which fragments are
          rendered, and is the only alpha_mode that can handle mixed transparent
          and opaque (alpha=1) fragments. It is therefore the most
          plug-and-play. If your semi-transparent object looks too noisy for
          your taste, try "over" instead. If you know your mesh is opaque, use
          "opaque" to avoid overdraw (and increase performance). The ``depth_write``
          defaults to True.
        * "weighted": Weighted blending, where the order of objects does not
          matter for the end-result. One use-case being order independent
          transparency (OIT).

        See ``dither_mode``, ``blend_mode``, and ``weighted_mode`` for more control
        over the respective alpha modes.

        The ``alpha_mode`` also affects how the renderer groups and sorts objects:

        * Objects with "opaque" or "dither" are rendered first, and are sorted front-to-back to avoid overdraw (to increase performance).
        * Objects with "blend" are rendered next, sorted back-to-front to increase the chance on correct blending order.
        * Objects with "weighted" are rendered last, not sorted (because order does not matter).
        * An object's `render_order` is leading, which may result in one of the above groups to occur more than once.

        The following values set ``alpha_mode`` to "blend", and ``blend_mode`` to the listed preset:

        * "over": use classic alpha blending using the *over operator* (the default for 'blend').
        * "add": use additive blending that adds the fragment color, multiplied by alpha.
        * "subtract": use subtractuve blending that removes the fragment color.
        * "multiply": use multiplicative blending that multiplies the fragment color.

        The following values set ``alpha_mode`` to "dither", and ``dither_mode`` to the listed preset:

        * "blue": use stochastic transparency with blue noise (the default for 'dither').
        * "bayer": use stochastic transparency with a Bayer pattern.

        """
        return self._store.alpha_mode

    @alpha_mode.setter
    def alpha_mode(self, alpha_mode: AlphaMode | str) -> None:
        if alpha_mode is None:
            alpha_mode = "blue"
        if not isinstance(alpha_mode, str):
            raise ValueError(
                f"alpha_mode is {alpha_mode!r} but expected one of {AlphaMode!r}"
            )

        # Presets
        real_alpha_mode = alpha_mode
        if alpha_mode in ["over", "add", "subtract", "multiply"]:
            self.blend_mode = alpha_mode
            real_alpha_mode = "blend"
        elif alpha_mode in ["blue", "bayer"]:
            self.dither_mode = alpha_mode
            real_alpha_mode = "dither"

        # Set alpha mode
        if real_alpha_mode in ["dither", "opaque", "blend", "weighted"]:
            self._store.alpha_mode = real_alpha_mode
        else:
            raise ValueError(
                f"Internal error, unexpect value for real_alpha_mode: {real_alpha_mode!r}."
            )

    def _gfx_get_alpha_mode_details(self):
        alpha_mode = self._store.alpha_mode
        if alpha_mode == "opaque":
            return {}
        elif alpha_mode == "blend":
            return self.blend_mode
        elif alpha_mode == "dither":
            return self.dither_mode
        elif alpha_mode == "weighted":
            return self.weighted_mode
        else:  # fail soft
            return {}

    @property
    def dither_mode(self) -> dict:
        """Details for when alpha_mode is 'dither'.

        The dict has the following fields:

        * "pattern": can be 'blue' for blue noise (default), 'white' for white
          noise, and 'bayer' for a Bayer pattern. The Bayer option mixes objects
          but not elements within objects.

        Can also be set using a preset string:

        * "blue": uses blue noise, resulting in fine-grained noise, and a
          pleasing result (the default).
        * "bayer": use a Bayer pattern resulting in a less noisy appearance.
          Different parts within an object have the same pattern so they don't
          "mix". Multiple objects have different seeds, so they do mix, albeit
          in a less consistent way than the stochastic transparency with blue
          noise.
        """
        return self._store.dither_mode.copy()

    @dither_mode.setter
    def dither_mode(self, dither_mode: dict | str):
        if dither_mode is None:
            dither_mode = "blue"
        preset = None
        if isinstance(dither_mode, str):
            preset = dither_mode.lower()
            if preset == "blue":
                dither_mode = {"pattern": "blue"}
            elif preset == "bayer":
                dither_mode = {"pattern": "bayer"}
            else:
                raise ValueError(
                    "Invalid preset for material.dither_mode: {dither_mode!r}"
                )

        keys = ["pattern"]
        defaults = {"pattern": "blue"}
        self._store.dither_mode = self._get_alpha_mode_sub_dict(
            "dither_mode", keys, defaults, dither_mode
        )
        if preset:
            self._store.dither_mode["preset"] = preset

    @property
    def blend_mode(self) -> dict:
        """Details for when alpha_mode is 'blend'.

        The dict has the following fields:

        * ``color_op``: the blend operation/equation, any value from ``wgpu.BlendOperation``. Default "add".
        * ``color_src``: source factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * ``color_dst``: destination factor, any value of ``wgpu.BlendFactor``. Mandatory.
        * ``color_constant``: represents the constant value of the constant blend color. Default black.
        * ``alpha_op``: as ``color_op`` but for alpha.
        * ``alpha_src``: as ``color_dst`` but for alpha.
        * ``alpha_dst``: as ``color_src`` but for alpha.
        * ``alpha_constant``: as ``color_constant`` but for alpha (default 0).

        Can also be set using a preset string:

        * "over": use classic alpha blending using the *over operator* (the default).
        * "add": use additive blending that adds the fragment color, multiplied by alpha.
        * "subtract": use subtractuve blending that removes the fragment color.
        * "multiply": use multiplicative blending that multiplies the fragment color.
        """
        return self._store.blend_mode.copy()

    @blend_mode.setter
    def blend_mode(self, blend_mode: dict | str):
        if blend_mode is None:
            blend_mode = "over"
        preset = None
        if isinstance(blend_mode, str):
            preset = blend_mode.lower()
            if preset == "over":
                blend_mode = {
                    "preset": "over",
                    "color_src": "src-alpha",
                    "color_dst": "one-minus-src-alpha",
                    "alpha_src": "one",
                    "alpha_dst": "one-minus-src-alpha",
                }
            elif preset == "add":
                blend_mode = {
                    "preset": "add",
                    "color_src": "src-alpha",
                    "color_dst": "one",
                    "alpha_src": "src-alpha",
                    "alpha_dst": "one",
                }
            elif preset == "subtract":
                blend_mode = {
                    "preset": "subtract",
                    "color_src": "zero",
                    "color_dst": "one-minus-src",
                    "alpha_src": "zero",
                    "alpha_dst": "one",
                }
            elif preset == "multiply":
                blend_mode = {
                    "color_src": "zero",
                    "color_dst": "src",
                    "alpha_src": "zero",
                    "alpha_dst": "src",
                }
            else:
                raise ValueError(
                    "Invalid preset for material.blend_mode: {blend_mode!r}"
                )

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
        self._store.blend_mode = self._get_alpha_mode_sub_dict(
            "blend_mode", keys, defaults, blend_mode
        )
        if preset:
            self._store.blend_mode["preset"] = preset

    @property
    def weighted_mode(self):
        """Details for when alpha_mode is 'weighted'.

        The dict has the following fields:

        * ``weight``: the weight factor as wgsl code. Default "alpha", which means use the color's alpha value.
        * ``alpha``: the used alpha value. Default "alpha", which means use as-is. Can e.g. be set to 1.0
          so that the alpha channel can be used as the weight factor, while the object is otherwise opaque.

        No preset names are currently defined.
        """
        return self._store.weighted_mode.copy()

    @weighted_mode.setter
    def weighted_mode(self, weighted_mode):
        if weighted_mode is None:
            weighted_mode = {}
        preset = None
        if isinstance(weighted_mode, str):
            # TODO: make a preset for this alpha_mode, for consistency?
            raise ValueError(
                "Invalid preset for material.weighted_mode: {weighted_mode!r}"
            )

        keys = ["weight", "alpha"]
        defaults = {"weight": "alpha", "alpha": "alpha"}
        self._store.weighted_mode = self._get_alpha_mode_sub_dict(
            "weighted_mode", keys, defaults, weighted_mode
        )
        if preset:
            self._store.weighted_mode["preset"] = preset

    def _get_alpha_mode_sub_dict(self, name, keys, default_dict, given_dict):
        if not isinstance(given_dict, dict):
            raise TypeError(f"material.{name} expects a dict, not {given_dict!r}")
        source_dict = given_dict.copy()
        result_dict = default_dict.copy()
        source_dict.pop("preset", None)
        try:
            for key in keys:
                if key in source_dict:
                    result_dict[key] = source_dict.pop(key)
        except KeyError as err:
            raise KeyError(
                f"material.{name} dict is missing field {err.args[0]!r}"
            ) from None
        if source_dict:
            raise ValueError(
                f"material.{name} dict contains invalid fields: {source_dict!r}"
            )
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
        * None: auto-determine (default): True if ``alpha_mode`` is "opaque" or "dither". Otherwise False.

        The auto-option provides good default behaviour for common use-case, but
        if you know what you're doing you should probably just set this value to
        True or False.
        """
        depth_write = self._store.depth_write
        if depth_write is None:
            depth_write = self._store.alpha_mode in ("opaque", "dither")
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
