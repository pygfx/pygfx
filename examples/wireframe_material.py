import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import Binding
from pygfx.resources import Buffer
from pygfx.renderers.wgpu.meshshader import WorldObjectShader


class WireframeMaterial(gfx.Material):
    uniform_type = dict(
        thickness="f4",
    )

    def __init__(self, *, thickness=1.0, **kwargs):
        super().__init__(**kwargs)
        self.thickness = thickness

    @property
    def thickness(self):
        return self.uniform_buffer.data["thickness"]

    @thickness.setter
    def thickness(self, thickness):
        self.uniform_buffer.data["thickness"] = thickness
        self.uniform_buffer.update_range(0, 1)

@gfx.renderers.wgpu.register_wgpu_render_function(gfx.WorldObject, WireframeMaterial)
class WireframeShader(WorldObjectShader):

    # Mark as render-shader (as opposed to compute-shader)
    type = "render"

    def get_resources(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        self["instanced"] = False

        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", material.uniform_buffer),
        }
        self.define_bindings(0, bindings)

        vertex_attributes = {}

        # to none-indexed geometry
        none_indexed_positions = None
        for face in geometry.indices.data:
            face_pos = geometry.positions.data[face]
            if none_indexed_positions is None:
                none_indexed_positions = face_pos
            else:
                none_indexed_positions = np.concatenate((none_indexed_positions, geometry.positions.data[face]))

        vertex_attributes["position"] = Buffer(none_indexed_positions)
        # centers = np.zeros_like(geometry.positions.data)
        # v_centers = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # for i in range(len(centers)):
        #     centers[i] = v_centers[i % 3]

        # vertex_attributes["center"] = Buffer(centers)

        self.define_vertex_buffer(vertex_attributes, instanced=self["instanced"])

        return {
            "index_buffer": None, # none-indexed geometry
            "vertex_buffers": list(vertex_attributes.values()),
            "instance_buffer": wobject.instance_infos if self["instanced"] else None,
            "bindings": {
                0: bindings,
            },
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        geometry = wobject.geometry

        n = geometry.indices.data.size
        n_instances = 1
        if self["instanced"]:
            n_instances = wobject.instance_buffer.nitems

        return {
            "indices": (n, n_instances),
            "render_mask": 3,
        }

    def get_code(self):
        return self.code_definitions() + self.code_vertex() + self.code_fragment()

    def code_vertex(self):
        return """
        struct Varyings {
            @builtin(position) position: vec4<f32>,
            @location(0) center: vec3<f32>,
        };

        @stage(vertex)
        fn vs_main(in: VertexInput) -> Varyings {
            var out: Varyings;
            let u_mvp = u_stdinfo.projection_transform * u_stdinfo.cam_transform * u_wobject.world_transform;
            out.position = u_mvp * vec4<f32>( in.position, 1.0 );

            let index = i32(in.vertex_index);
            if(index % 3 == 0) {
                out.center = vec3<f32>( 1.0, 0.0, 0.0 );
            }else if(index % 3 == 1) {
                out.center = vec3<f32>( 0.0, 1.0, 0.0 );
            }else {
                out.center = vec3<f32>( 0.0, 0.0, 1.0 );
            }

            // out.center = in.center;
            return out;
        }
        """

    def code_fragment(self):
        return """
        struct FragmentOutput {
            @location(0) color: vec4<f32>,
            @location(1) pick: vec4<u32>,
        };


        @stage(fragment)
        fn fs_main(in: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {
            var out: FragmentOutput;
            let afwidth = fwidth( in.center.xyz );
            let thickness = u_material.thickness;

            let edge3 = smoothStep( ( thickness - 1.0 ) * afwidth, thickness * afwidth, in.center.xyz );

            let edge = 1.0 - min( min( edge3.x, edge3.y ), edge3.z );

            if ( edge > 0.01 ) {
                if ( is_front ) {
                    out.color = vec4<f32>( 0.9, 0.9, 1.0, 1.0 );
                }else {
                    out.color = vec4<f32>( 0.4, 0.4, 0.5, 0.5);
                }
            } else {
                discard;
            }

            //let color = vec4<f32>( 0.9, 0.9, 1.0, edge );
            //out.color = color;
            return out;
        }
        """


# %% Setup scene

renderer = gfx.WgpuRenderer(WgpuCanvas(size=(640, 480)))
camera = gfx.PerspectiveCamera(45, 640 / 480, 0.1, 100)
camera.position.z = 10

# camera = gfx.OrthographicCamera(8, 6, -1, 1)

# g = gfx.torus_knot_geometry(1, 0.3, 128, 32)

g = gfx.sphere_geometry(1, 16)

mesh1 = gfx.Mesh(g, WireframeMaterial(thickness=0.3))
mesh1.position.x = -3

mesh2 = gfx.Mesh(g, WireframeMaterial())
mesh2.position.x = 0

mesh3 = gfx.Mesh(g, gfx.MeshPhongMaterial(wireframe=True))
mesh3.position.x = 3

scene = gfx.Scene()
scene.add(gfx.DirectionalLight())
scene.add(gfx.AmbientLight())

scene.add(mesh1)
scene.add(mesh2)
scene.add(mesh3)

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
