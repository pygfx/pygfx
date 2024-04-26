"""
Depth Material
==============

A custom material for drawing geometry by depth
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import Binding, register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.meshshader import MeshShader


class DepthMaterial(gfx.MeshBasicMaterial):
    pass


@register_wgpu_render_function(gfx.Mesh, DepthMaterial)
class DepthShader(MeshShader):
    # Mark as render-shader (as opposed to compute-shader)
    type = "render"

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry

        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
        }
        bindings[2] = Binding(
            "s_indices", "buffer/read_only_storage", geometry.indices, "VERTEX"
        )
        bindings[3] = Binding(
            "s_positions", "buffer/read_only_storage", geometry.positions
        )
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        # We draw triangles, no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        geometry = wobject.geometry
        n = geometry.indices.data.size
        return {
            "indices": (n, 1),
            "render_mask": 3,
        }

    def get_code(self):
        return """
        {$ include 'pygfx.std.wgsl' $}

        struct VertexInput {
            @builtin(vertex_index) vertex_index : u32,
        };

        @vertex
        fn vs_main(in: VertexInput) -> Varyings {

            let vertex_index = i32(in.vertex_index);
            let face_index = vertex_index / 3;
            var sub_index = vertex_index % 3;
            let ii = load_s_indices(face_index);
            let i0 = i32(ii[sub_index]);

            let position = load_s_positions(i0);
            let u_mvp = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform;
            let pos = u_mvp * vec4<f32>(position, 1.0);

            var varyings: Varyings;
            varyings.position = vec4<f32>(pos);
            return varyings;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;
            let depth = 1.0 - varyings.position.z; // Invert depth - could also do logarithmic depth
            out.color = vec4<f32>(depth);
            return out;
        }
        """


# Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas(size=(640, 480)))

camera = gfx.PerspectiveCamera(45, 640 / 480, depth_range=(8, 12))
camera.local.z = 10
camera.show_pos((0, 0, 0))

t = gfx.Mesh(gfx.torus_knot_geometry(1, 0.3, 128, 32), DepthMaterial())

scene = gfx.Scene()
scene.add(t)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
