"""
.. currentmodule:: pygfx.renderers.wgpu

The ``EffectPass`` is the base class for implementing full screen post-processing effects.
This can be subclasses to create custom effects. A few builtin effects are also available.


.. autosummary::
    :toctree: _autosummary/renderers/wgpu/engine/effectpasses
    :template: ../_templates/custom_layout.rst

    EffectPass
    CopyPass
    PPAAPass
    FXAAPass
    DDAAPass
    NoisePass
    DepthPass
    FogPass

"""

import time

import wgpu

from ....utils import array_from_shadertype
from ....utils.color import Color
from ..shader.bindings import BindingDefinitions
from ..shader.templating import apply_templating
from .utils import GpuCache, hash_from_value
from .shared import get_shared
from .binding import Binding


# This cache enables sharing some gpu objects between code that uses
# full-quad shaders. The gain here won't be large in general, but can
# still be worthwhile in situations where many canvases are created,
# such as in Jupyter notebooks, or e.g. when generating screenshots.
# It should also reduce the required work during canvas resizing.
FULL_QUAD_CACHE = GpuCache("full_quad_objects")


FULL_QUAD_SHADER = """
    struct VertexInput {
        @builtin(vertex_index) index: u32,
    };
    struct Varyings {
        @location(0) texCoord: vec2<f32>,
        @builtin(position) position: vec4<f32>,
    };

    @vertex
    fn vs_main(in: VertexInput) -> Varyings {
        var positions = array<vec2<f32>,4>(
            vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
        );
        let pos = positions[in.index];
        var varyings: Varyings;
        varyings.texCoord = vec2<f32>(pos.x, 1.0 - pos.y);
        varyings.position = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
        return varyings;
    }
"""


def create_full_quad_pipeline(targets, binding_layout, fragment_code):
    """Low-level support for a full-quad pipeline."""
    device = get_shared().device

    # Get bind group layout
    key1 = hash_from_value(binding_layout)
    bind_group_layout = FULL_QUAD_CACHE.get(key1)
    if bind_group_layout is None:
        bind_group_layout = device.create_bind_group_layout(entries=binding_layout)
        FULL_QUAD_CACHE.set(key1, bind_group_layout)

    # Get render pipeline
    key2 = hash_from_value([bind_group_layout, targets, fragment_code])
    render_pipeline = FULL_QUAD_CACHE.get(key2)
    if render_pipeline is None:
        wgsl = FULL_QUAD_SHADER + fragment_code
        shader_module = device.create_shader_module(code=wgsl)

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
                "strip_index_format": wgpu.IndexFormat.uint32,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": targets,
            },
        )

        # Bind shader module object to the lifetime of the pipeline object
        render_pipeline._gfx_module = shader_module

        FULL_QUAD_CACHE.set(key2, render_pipeline)

    return render_pipeline


# ----- Class hierachy


class FullQuadPass:
    """
    A base class for rendering a full quad, with support for uniforms.
    The only current subclass is the EffectPass. The design is such that it supports
    as many source and target textures as you want, so effects can e.g. have a texture
    as a property.
    """

    uniform_type = dict()

    wgsl = ""

    def __init__(self):
        self._device = get_shared().device

        self._uniform_data = array_from_shadertype(self.uniform_type)
        self._wgpu_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        bindings_definition = BindingDefinitions()
        bindings_definition.define_binding(
            0, 0, Binding("u_effect", "buffer/uniform", self._uniform_data.dtype)
        )
        self._uniform_binding_definition = bindings_definition.get_code()
        self._uniform_binding_entry = {
            "binding": 0,
            "resource": {
                "buffer": self._wgpu_buffer,
                "offset": 0,
                "size": self._wgpu_buffer.size,
            },
        }

        self._wgpu_sampler = self._device.create_sampler(
            min_filter="linear", mag_filter="linear"
        )
        self._sampler_binding_definition = """
            @group(0) @binding(1)
            var texSampler: sampler;
        """
        self._sampler_binding_entry = {"binding": 1, "resource": self._wgpu_sampler}

        self._render_pipeline = None
        self._render_pipeline_hash = None
        self._template_vars = {}
        self._template_vars_changed = True

    def _set_template_var(self, **kwargs):
        for name, value in kwargs.items():
            if value != self._template_vars.get(name, "stubvaluethatsnotit"):
                self._template_vars_changed = True  # causes a shader recompile
        self._template_vars.update(kwargs)

    def render(
        self,
        command_encoder,
        **texture_views,
    ):
        """Render the source into the destination, applying the postprocessing effect."""

        source_textures = {}
        target_textures = []

        for name, tex in texture_views.items():
            if not isinstance(tex, wgpu.GPUTextureView):
                raise TypeError(f"FullQuadPass expected a texture view, not {tex!r}")
            if name.startswith("target"):
                target_textures.append(tex)
            else:
                source_textures[name] = tex
        if not target_textures:
            raise RuntimeError(
                "FullQuadPass needs at least one target texture (prefixed with 'target')."
            )

        # Make sure we have the render_pipeline
        render_pipeline_hash = (
            tuple(source_textures.keys()),
            tuple(tex.texture.format for tex in target_textures),
        )
        if (
            self._template_vars_changed
            or render_pipeline_hash != self._render_pipeline_hash
        ):
            self._template_vars_changed = False
            self._render_pipeline_hash = render_pipeline_hash
            self._render_pipeline = self._create_pipeline(*render_pipeline_hash)

        # Update uniforms
        self._device.queue.write_buffer(
            self._wgpu_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

        # Ready to go!
        return self._render(command_encoder, source_textures.values(), target_textures)

    def _render(self, command_encoder, source_textures, target_textures):
        # Create bind group. This is very light and can be done every time.
        # Chances are we get new views on every call anyway.
        bind_group_entries = [self._uniform_binding_entry, self._sampler_binding_entry]
        for i, tex in enumerate(source_textures, 2):
            bind_group_entries.append({"binding": i, "resource": tex})
        bind_group = self._device.create_bind_group(
            layout=self._render_pipeline.get_bind_group_layout(0),
            entries=bind_group_entries,
        )

        # Create attachments
        color_attachments = []
        for tex in target_textures:
            color_attachments.append(
                {
                    "view": tex,
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            )

        render_pass = command_encoder.begin_render_pass(
            color_attachments=color_attachments,
            depth_stencil_attachment=None,
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 99)
        render_pass.draw(4, 1)
        render_pass.end()

    def _create_pipeline(self, source_names, target_formats):
        binding_layout = []
        definitions_code = ""

        # Uniform buffer
        binding_layout.append(
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        )
        definitions_code += self._uniform_binding_definition

        # Sampler
        binding_layout.append(
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            }
        )
        definitions_code += self._sampler_binding_definition

        # Source textures
        for i, name in enumerate(source_names, 2):
            sample_type = wgpu.TextureSampleType.float
            wgsl_type = "texture_2d<f32>"
            if "depth" in name.lower():
                sample_type = wgpu.TextureSampleType.depth
                wgsl_type = "texture_depth_2d"
            binding_layout.append(
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": sample_type,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                        "multisampled": False,
                    },
                }
            )
            definitions_code += f"""
                @group(0) @binding({i})
                var {name}: {wgsl_type};
            """

        # Render targets
        targets = []
        for format in target_formats:
            targets.append(
                {
                    "format": format,
                    "blend": {
                        "alpha": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                        "color": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                    },
                }
            )

        wgsl = definitions_code
        wgsl += apply_templating(self.wgsl, **self._template_vars)
        return create_full_quad_pipeline(targets, binding_layout, wgsl)


class EffectPass(FullQuadPass):
    """
    Base class to do post-processing effect passes, converting one image into another.
    """

    # TODO: use templating and keep a dict with template variables. That we we can e.g. scale the support of a Gaussian blur effect properly.

    USES_DEPTH = False
    """Overloadable class attribute to state whether bindings to the depth buffer are needed.
    """

    uniform_type = dict(
        time="f4",
    )
    """Overloadable class attribute that defines the structure of the uniform struct.

    In a subclass you should use something like this::

        uniform_type = dict(
            EffectPass.uniform_type,
            color="4xf4",
            strength="f4",
        )
    """

    wgsl = "EffectPass_needs_to_be_subclassed(and_its_wgsl_attr_overloaded);"
    """Overloadable class attribute that contains the WGSL shading code for the fragment shader.

    Use the following template::

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {

            // Available variables:
            // colorTex - the texture containing the rendered image, or the previous effect pass.
            // depthTex - the texture containing the renderd depth values.
            // texSampler - a sampler to use for the above.
            // varyings.position - the position in physical pixels (a vec4f).
            // varyings.texCoord - the coordinate in the textures (a vec2f).
            // u_effect.time - the current time in seconds, changes each frame.
            // u_effect.xx - whatever uniforms you added.

            // Calculate the pixel index, e.g. if you want to use textureLoad().
            let texIndex = vec2i(varyings.position.xy);

            // To simply copy the image:
            return textureSample(colorTex, texSampler, varyings.texCoord);
        }
    """

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def render(
        self,
        command_encoder,
        color_tex,
        depth_tex,
        target_tex,
    ):
        """Render the pass using the provided textures.

        If ``USES_DEPTH`` is False, the ``depth_tex`` is ignored (and may be set to None).
        """
        # Set uniforms
        self._uniform_data["time"] = time.perf_counter()

        # Compose kwargs containing the textures. Target textures are prefixed with 'target'
        kwargs = dict(colorTex=color_tex, targetTex=target_tex)
        if self.USES_DEPTH:
            kwargs["depthTex"] = depth_tex

        super().render(command_encoder, **kwargs)


class CopyPass(EffectPass):
    """
    Simple pass that does nothing but copy the texture over, using linear interpolation if the texture size does not match.
    Mostly included for testing.
    """

    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            return textureSample(colorTex, texSampler, varyings.texCoord);
        }
        """


class OutputPass(EffectPass):
    """
    Render from one texture into another, taking size difference into account. Applying gamma on the way.
    """

    # Note: This class is not public, but used internally by the renderer to copy the final result to the target texture. So technically not so much an 'effect'.

    uniform_type = dict(
        EffectPass.uniform_type,
        gamma="f4",
    )

    wgsl = "{$ include 'pygfx.ssaa.wgsl' $}"

    def __init__(self, *, gamma=1.0, filter="mitchell"):
        super().__init__()
        self.gamma = gamma
        self.filter = filter
        self._set_template_var(
            extraKernelSupport=None,  # for testing
            optCorners=True,  # optimization: drop corners in kernels larger than 6x6
            optScale2=True,  # optimization: use 12-tap filters for cubic kernels when scaleFactor == 2
            gamma="u_effect.gamma",  # let gamma use the uniform buffer
        )

    def render(
        self,
        command_encoder,
        color_tex,
        depth_tex,
        target_tex,
    ):
        """Render the (internal) result of the renderer to a texture view."""

        # Get factor from source to destination. A value < 1 means the source is smaller; we're upsampling.
        factor_x = color_tex.size[0] / target_tex.size[0]
        factor_y = color_tex.size[1] / target_tex.size[1]
        factor = (factor_x + factor_y) / 2

        self._set_template_var(scaleFactor=float(factor))

        return super().render(command_encoder, color_tex, depth_tex, target_tex)

    @property
    def gamma(self):
        """The gamma correction to apply."""
        return float(self._uniform_data["gamma"])

    @gamma.setter
    def gamma(self, gamma):
        self._uniform_data["gamma"] = float(gamma)

    @property
    def filter(self):
        """The type of filter to use."""
        return self._template_vars["filter"]

    @filter.setter
    def filter(self, filter):
        filter = filter.lower()
        filters = {
            "nearest",
            "box",
            "linear",
            "tent",
            "disk",
            "mitchell",
            "bspline",
            "catmull",
        }
        if filter not in filters:
            raise ValueError(
                f"Unknown OutputPass filter {filter!r}, must be one of {set(filters)}"
            )
        self._set_template_var(filter=filter)


# ----- Builtin effects

# Creating effects is fun! Some ideas:
#
# BloomPass, FilmPass, GlitchPass, GodRayPass, OutlinePass, AmbientOcclusionPass, FogPass, ...
# GaussXPass, GaussYPass, GaussDxPass, GaussDyPass, SobelPass, LaplacianPass, SharpeningPass, ...
# Color grading, color conversions, depth of field, patterns, pixelize, tone mapping, texture overlay, ...
#
# See https://github.com/pmndrs/postprocessing and ThreeJS code for implementations.


class PPAAPass(EffectPass):
    """Base class for post-processing anti-aliasing to help the renderer detect these."""


class FXAAPass(PPAAPass):
    """An effect pass implementing Fast approximate anti-aliasing.

    FXAA is a well known method for post-processing antialiasing.
    This is version 3.11.
    """

    wgsl = "{$ include 'pygfx.fxaa3.wgsl' $}"


class DDAAPass(PPAAPass):
    """An effect pass implementing Directional Diffusion anti-aliasing.

    DDAA produces better results than FXAA for near-diagonal lines, at the same performance.
    It estimates the direction of the edge, and then diffuses (i.e. smoothes) in that direction.
    For near-horizontal and near-vertical a technique similar to FXAA is used.
    """

    wgsl = "{$ include 'pygfx.ddaa2.wgsl' $}"

    def __init__(self, *, max_edge_iters=5):
        super().__init__()
        self.max_edge_iters = max_edge_iters

    @property
    def max_edge_iters(self):
        """The maximum number of iters (of 3 samples) to search along an edge.

        Default 5 (i.e. search 15 pixels along an edge). Set higger for prettier
        edges, or lower for more performance.
        """
        return len(self._template_vars["EDGE_STEP_LIST"])

    @max_edge_iters.setter
    def max_edge_iters(self, max_edge_iters):
        # The default EDGE_STEP_LIST is [3, 3, 3, 3, 3]
        edge_step_list = [3] * int(max_edge_iters)
        self._set_template_var(EDGE_STEP_LIST=edge_step_list)


class NoisePass(EffectPass):
    """An effect pass that adds noise."""

    uniform_type = dict(
        EffectPass.uniform_type,
        noise="f4",
    )

    wgsl = """
        {$ include 'pygfx.noise.wgsl' $}

        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texCoord = varyings.texCoord;
            let texIndex = vec2i(varyings.position.xy);
            let noise = random(texCoord.x * texCoord.y * u_effect.time);
            let color = textureLoad(colorTex, texIndex, 0);
            return color + noise * u_effect.noise;
        }
    """

    def __init__(self, noise=0.1):
        super().__init__()
        self.noise = noise

    @property
    def noise(self):
        """The strength of the noise."""
        return float(self._uniform_data["noise"])

    @noise.setter
    def noise(self, noise):
        self._uniform_data["noise"] = float(noise)


class DepthPass(EffectPass):
    """An effect that simply renders the depth. Mostly for debugging purposes."""

    USES_DEPTH = True

    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);
            let depth = textureLoad(depthTex, texIndex, 0);
            return vec4f(depth, depth, depth, 1.0);
        }
    """


class FogPass(EffectPass):
    """An effect pass that adds fog to the full image, using the depth buffer."""

    USES_DEPTH = True

    uniform_type = dict(
        EffectPass.uniform_type,
        color="4xf4",
        power="f4",
    )

    wgsl = """
        @fragment
        fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
            let texIndex = vec2i(varyings.position.xy);
            let raw_depth = textureLoad(depthTex, texIndex, 0);
            let depth = pow(raw_depth, u_effect.power);

            let color = textureLoad(colorTex, texIndex, 0);
            let depth_color = u_effect.color;

            return mix(color, depth_color, depth);
        }
    """

    def __init__(self, color="#fff", power=1.0):
        super().__init__()
        self.color = color
        self.power = power

    @property
    def color(self):
        """The color of the fog."""
        return Color(self._uniform_data["color"])

    @color.setter
    def color(self, color):
        color = Color(color)
        self._uniform_data["color"] = color

    @property
    def power(self):
        """The power to apply to the depth value. Using a value < 1 brings the range more towards the camera."""
        return float(self._uniform_data["power"])

    @power.setter
    def power(self, power):
        self._uniform_data["power"] = float(power)
