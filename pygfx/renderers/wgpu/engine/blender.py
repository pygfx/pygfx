"""
Defines the classes for the different render passes and the blender
objects that contain them. A blender becomes part of the renderstate
object.
"""

import wgpu  # only for flags/enums

from ....utils.enums import RenderMask
from .shared import get_shared
from .renderer import WgpuRenderer

# Notes:
# - The user code provides color as-is in rgba.
# - In the get_fragment_output logic defined in the shaders here, the color
#   is pre-multiplied with the alpha.
# - All fixed-pipeling blending options assume premultiplied alpha.


standard_texture_des = {
    "sample_type": wgpu.TextureSampleType.unfilterable_float,
    "view_dimension": wgpu.TextureViewDimension.d2,
    "multisampled": False,
}


def blend_dict(src_factor, dst_factor, operation):
    return {
        "operation": operation,
        "src_factor": src_factor,
        "dst_factor": dst_factor,
    }


# %%%%%%%%%%  Define passes


class BasePass:
    """The base pass class, defining and documenting the API that a pass must provide."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        """Get the list of fragment targets for device.create_render_pipeline()."""
        # The result affects the wobject's pipeline.
        # Returning [] prevents this pipeline from being created.
        return []

    def get_color_attachments(self, blender):
        """Get the list of color_attachments for command_encoder.begin_render_pass()."""
        # The result affects the rendering, but not the wobject's pipeline.
        # Returning [] prevents this pass from running.
        return []

    def get_depth_descriptor(self, blender):
        """Get dict that has the depth-specific fields for the depth_stencil field
        in device.create_render_pipeline().
        """
        return {}

    def get_depth_attachment(self, blender):
        """Get a dict that has the depth-specific fields for the
        depth_stencil_attachment argument of command_encoder.begin_render_pass().
        """
        return {}

    def get_shader_code(self, blender):
        """Get the fragment-write shader code.

        Notes:

        * This code gets injected into the shader, so the material shaders
          can use get_fragment_output.
        * This code should define FragmentOutput and get_fragment_output.
        """
        return ""


class OpaquePass(BasePass):
    """A pass that renders opaque fragments with depth testing, while
    discarting transparent fragments. This functions as the first pass
    in all multi-pass blenders.
    """

    render_mask = RenderMask.opaque
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        return [
            {
                "format": blender.color_format,
                "blend": {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                },
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
            },
        ]

    def get_color_attachments(self, blender):
        color_load_op = pick_load_op = wgpu.LoadOp.load
        if blender.color_clear:
            blender.color_clear = False
            color_load_op = wgpu.LoadOp.clear
        if blender.pick_clear:
            blender.pick_clear = False
            pick_load_op = wgpu.LoadOp.clear

        return [
            {
                "view": blender.color_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": color_load_op,
                "store_op": wgpu.StoreOp.store,
            },
            {
                "view": blender.pick_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": pick_load_op,
                "store_op": wgpu.StoreOp.store,
            },
        ]

    def get_depth_descriptor(self, blender):
        return {
            "format": blender.depth_format,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        }

    def get_depth_attachment(self, blender):
        depth_load_op = wgpu.LoadOp.load
        if blender.depth_clear:
            blender.depth_clear = False
            depth_load_op = wgpu.LoadOp.clear
        return {
            "view": blender.depth_view,
            "depth_clear_value": 1.0,
            "depth_load_op": depth_load_op,
            "depth_store_op": wgpu.StoreOp.store,
        }

    def get_shader_code(self, blender):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(position: vec4<f32>, color: vec4<f32>) -> FragmentOutput {
            if (color.a < 1.0 - ALPHA_COMPARE_EPSILON ) { discard; }
            var out : FragmentOutput;
            out.color = vec4<f32>(color.rgb, 1.0);
            return out;
        }
        """


class SimpleSinglePass(OpaquePass):
    """A pass that blends opaque and transparent fragments in a single pass."""

    render_mask = RenderMask.opaque | RenderMask.transparent
    write_pick = True

    def get_color_descriptors(self, blender, material_write_pick, blending):
        bf, bo = wgpu.BlendFactor, wgpu.BlendOperation
        if isinstance(blending, str):
            if blending == "no":
                blend = {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                }
            elif blending == "normal":
                blend = {
                    "alpha": blend_dict(bf.one, bf.one_minus_src_alpha, bo.add),
                    "color": blend_dict(bf.src_alpha, bf.one_minus_src_alpha, bo.add),
                }
            elif blending == "add":
                blend = {
                    "alpha": blend_dict(bf.one, bf.one, bo.add),
                    "color": blend_dict(bf.one, bf.one, bo.add),
                }
            elif blending == "subtract":
                # todo: correct?
                blend = {
                    "alpha": blend_dict(bf.one, bf.one, bo.subtract),
                    "color": blend_dict(bf.one, bf.one, bo.subtract),
                }
            elif blending == "dither":
                blend = {
                    "alpha": blend_dict(bf.one, bf.zero, bo.add),
                    "color": blend_dict(bf.one, bf.zero, bo.add),
                }

        return [
            {
                "format": blender.color_format,
                "blend": blend,
                "write_mask": wgpu.ColorWrite.ALL,
            },
            {
                "format": blender.pick_format,
                "blend": None,
                "write_mask": wgpu.ColorWrite.ALL if material_write_pick else 0,
            },
        ]

    def get_shader_code(self, blender, blending):
        # Take depth into account, but don't treat transparent fragments differently

        if blending == "no":
            outlines = "out.color = vec4<f32>(color.rgb, 1.0);"
        elif blending == "normal":
            outlines = "out.color = vec4<f32>(color.rgb, color.a);"
        elif blending == "dither":
            # We want the seed for the random function to be such that the result is
            # deterministic, so that rendered images can be visually compared. This
            # is why the object-id should not be used. Using the xy ndc coord is a
            # no-brainer seed. Using only these will give an effect often observed
            # in games, where the pattern is "stuck to the screen". We also seed with
            # the depth, since this covers *a lot* of cases, e.g. different objects
            # behind each-other, as well as the same object having different parts
            # at the same screen pixel. This only does not cover cases where objects
            # are exactly on top of each other. Therefore we use rgba as another seed.
            # So the only case where the same pattern may be used for different
            # fragments if an object is at the same depth and has the same color.
            outlines = """
            let seed1 = position.x * position.y * position.z;
            let seed2 = color.r * 0.12 + color.g * 0.34 + color.b * 0.56 + color.a * 0.78;
            let rand = random2(vec2<f32>(seed1, seed2));
            if ( color.a < 1.0 - ALPHA_COMPARE_EPSILON && color.a < rand ) { discard; }
            out.color = vec4<f32>(color.rgb, 1.0);  // fragments that pass through are opaque
            """.strip()
        else:
            outlines = "out.color = vec4<f32>(color.rgb, color.a);"

        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };
        fn get_fragment_output(position: vec4<f32>, color: vec4<f32>) -> FragmentOutput {
            var out : FragmentOutput;
            OUTLINES
            return out;
        }
        """.replace("OUTLINES", outlines)


# %%%%%%%%%% Define blenders


class BaseFragmentBlender:
    """Manage how fragments are blended and end up in the final target.
    Each renderer has one blender object.
    """

    # Each blender should define a unique name so that it may be
    # correctly registered in the renderer.
    name = None
    passes = []

    def __init__(self):
        self.device = get_shared().device

        # The size (2D in pixels) of the frame textures.
        self.size = (0, 0)

        # Objects for the combination pass
        self._combine_pass_pipeline = None
        self._combine_pass_bind_group = None

        # A dict that contains the metadata for all render targets.
        self._texture_info = {}

        # The below targets are always present, and the renderer expects their
        # format, texture, and view to be present.
        # These contribute to 4+4+8 = 16 bytes per pixel

        usg = wgpu.TextureUsage

        # The color texture is in srgb, because in the shaders we work with physical
        # values, but we want to store as srgb to make effective use of the available bits.
        self._texture_info["color"] = (
            wgpu.TextureFormat.rgba8unorm_srgb,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC | usg.TEXTURE_BINDING,
        )

        # The depth buffer is 32 bit - we need that precision.
        # Note that there is also depth32float-stencil8, but it needs the
        # (webgpu) extension with the same name.
        self._texture_info["depth"] = (
            wgpu.TextureFormat.depth32float,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

        # The pick texture has 4 16bit channels, adding up to 64 bits.
        # These bits are divided over the pick data, e.g. 20 for the wobject id.
        self._texture_info["pick"] = (
            wgpu.TextureFormat.rgba16uint,
            usg.RENDER_ATTACHMENT | usg.COPY_SRC,
        )

    def clear(self):
        """Clear the buffers."""
        for key in self._texture_info.keys():
            setattr(self, key + "_clear", True)

    def clear_depth(self):
        """Clear the deph buffer only."""
        self.depth_clear = True

    def ensure_target_size(self, size):
        """If necessary, resize render-textures to match the target size."""

        assert len(size) == 2
        size = size[0], size[1]
        if size == self.size:
            return

        # Set new size
        self.size = size
        tex_size = (*size, 1)

        # Any bind group is now invalid because they include source textures.
        self._combine_pass_bind_group = None

        # Recreate internal textures
        for name, (format, usage) in self._texture_info.items():
            wgpu_texture = self.device.create_texture(
                size=tex_size, usage=usage, dimension="2d", format=format
            )
            setattr(self, name + "_format", format)
            setattr(self, name + "_tex", wgpu_texture)
            setattr(self, name + "_view", wgpu_texture.create_view())
            setattr(self, name + "_clear", True)

    # The five methods below represent the API that the render system uses.

    def get_color_descriptors(self, pass_index, material_write_pick, material_blending):
        return self.passes[pass_index].get_color_descriptors(
            self, material_write_pick, material_blending
        )

    def get_color_attachments(self, pass_index):
        return self.passes[pass_index].get_color_attachments(self)

    def get_depth_descriptor(self, pass_index, depth_test=True, depth_write=True):
        des = self.passes[pass_index].get_depth_descriptor(self)
        if not depth_test:
            des["depth_compare"] = wgpu.CompareFunction.always
        if not depth_write:
            des["depth_write_enabled"] = False
        return {
            **des,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
            "stencil_front": {},  # use defaults
            "stencil_back": {},  # use defaults
        }

    def get_depth_attachment(self, pass_index):
        return self.passes[pass_index].get_depth_attachment(self)
        # We don't use the stencil yet, but when we do, we will also have to specify
        # "stencil_read_only", "stencil_load_op", and "stencil_store_op"

    def get_shader_kwargs(self, pass_index, material_blending):
        return {
            "blending_code": self.passes[pass_index].get_shader_code(
                self, material_blending
            ),
            "write_pick": self.passes[pass_index].write_pick,
        }

    def get_pass_count(self):
        """Get the number of passes for this blender."""
        return len(self.passes)

    def perform_combine_pass(self):
        """Perform a render-pass to combine any multi-pass results, if needed."""
        return []


@WgpuRenderer._register_blend_mode
class Ordered1FragmentBlender(BaseFragmentBlender):
    """A minimal fragment blender that uses the classic alpha blending
    equation, without differentiating between opaque and transparent
    objects. This is a common approach for applications using a single
    pass. Order dependent.
    """

    name = "ordered1"

    passes = [SimpleSinglePass()]
