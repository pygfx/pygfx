"""
Validate viewport offset
========================

Draw the same object into two viewports of the same size, at different offsets
on the canvas, and check that it comes out the same in both.

The triangle's colour is decided by where each fragment sits *inside its
viewport*. A fragment shader has to work that out from ``@builtin(position)``,
which is measured from the framebuffer's top-left corner, and the viewport's
own position, which reaches the shader as ``u_stdinfo.physical_offset``.

The vertex stage places the triangle in NDC, and NDC is already relative to the
viewport, so the geometry is identical wherever the viewport goes. Only the
fragment stage can tell the two viewports apart. Both halves of this image must
therefore be identical; if they are not, the fragment stage has lost track of
where its viewport starts.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import wgpu
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    register_wgpu_render_function,
)

SIZE = 200  # each viewport is SIZE x SIZE logical pixels


class ScreenPositionTriangle(gfx.WorldObject):
    pass


class ScreenPositionMaterial(gfx.Material):
    pass


@register_wgpu_render_function(ScreenPositionTriangle, ScreenPositionMaterial)
class ScreenPositionShader(BaseShader):
    type = "render"

    def get_bindings(self, wobject, shared, scene):
        bindings = {0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer)}
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        return {"indices": (3, 1)}

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(-0.92, -0.92),
                vec2<f32>( 0.92, -0.92),
                vec2<f32>( 0.00,  0.92),
            );
            var varyings: Varyings;
            varyings.position = vec4<f32>(positions[index], 0.0, 1.0);
            return varyings;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            // The viewport's size and its origin, both in physical pixels.
            let size = u_stdinfo.physical_size.xy;
            let offset = u_stdinfo.physical_offset.xy;

            // position.xy is measured from the framebuffer's corner, so the
            // offset must come off to make it relative to this viewport.
            let here = varyings.position.xy - offset;

            // Two things that depend on the fragment's place in the viewport:
            // a gradient over the full width and height, and rings centred on
            // the middle. Both are wrong if the offset is wrong.
            let uv = here / size;
            let centred = (here - 0.5 * size) / (0.5 * size.y);
            let rings = 0.5 + 0.5 * sin(length(centred) * 22.0);

            var out: FragmentOutput;
            out.color = vec4<f32>(uv.x, uv.y, rings, 1.0);
            return out;
        }
        """


canvas = RenderCanvas(size=(2 * SIZE, SIZE))
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.ppaa = "none"  # keep the comparison exact

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#101418"))
scene.add(ScreenPositionTriangle(None, ScreenPositionMaterial()))

camera = gfx.NDCCamera()


def animate():
    # Two viewports of the same size, at different offsets. The left one starts
    # at the canvas corner, where the framebuffer and viewport frames coincide;
    # the right one does not, which is what makes this a test.
    renderer.render(scene, camera, flush=False, rect=(0, 0, SIZE, SIZE), clear=True)
    renderer.render(scene, camera, flush=False, rect=(SIZE, 0, SIZE, SIZE))
    renderer.flush()


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    loop.run()
