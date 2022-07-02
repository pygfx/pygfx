import numpy as np
import imageio
import time
import math
import wgpu
import pygfx as gfx
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader
from colorsys import hls_to_rgb


class Skybox(gfx.WorldObject):
    
    def __init__(self, material=None):
        super().__init__(material=material)
        self.box = gfx.box_geometry(2, 2, 2)


class SkyboxMaterial(gfx.Material):
    
    def __init__(self, *, map, **kwargs):
        super().__init__(**kwargs)
        self.map = map


class SkyboxShader(WorldObjectShader):
    def get_code(self):
        return (
            self.get_definitions()
            + self.common_functions()
            + self.vertex_shader()
            + self.fragment_shader()
        )

    def vertex_shader(self):
        return """
        struct SkyOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec3<f32>,
        };

        struct VertexIn {
            @location(0) position: vec3<f32>,
        };

        @stage(vertex)
        fn vs_main(in: VertexIn) -> SkyOutput {
            var result: SkyOutput;
            result.uv = in.position.xyz;
            var view_matrix = u_stdinfo.cam_transform;
            view_matrix[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            let u_mvpNoscale = u_stdinfo.projection_transform * view_matrix * u_wobject.world_transform;
            result.position = u_mvpNoscale * vec4<f32>( in.position, 1.0 );
            result.position.z = 0.0;
            return result;
        }

    """

    def fragment_shader(self):
        return """
        @stage(fragment)
        fn fs_main(vertex: SkyOutput) -> FragmentOutput {
            let color = textureSample(r_texture, r_sampler, vertex.uv);
            var out: FragmentOutput;
            out.color = color;
            return out;
        }

    """


@gfx.renderers.wgpu.register_wgpu_render_function(Skybox, SkyboxMaterial)
def skybox_render_function(render_info):

    shader = SkyboxShader(render_info)

    wobject = render_info.wobject
    material = wobject.material
    geometry = wobject.box

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "indices", None)
    n = geometry.indices.data.size

    # We're using storage buffers for everything; no vertex nor index buffers.
    vertex_buffers = []
    index_buffer = geometry.indices
    shader["vertex_attributes"] = []

    vertex_buffers.append(geometry.positions)
    shader["vertex_attributes"].append(("position", "vec3<f32>"))
    

    binding = Binding("u_stdinfo", "buffer/uniform", render_info.stdinfo_uniform)
    shader.define_binding(0, 0, binding)

    binding1 = Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer)
    shader.define_binding(0, 1, binding1)

    binding2 = Binding("r_texture", "texture/auto", material.map, "FRAGMENT")
    shader.define_binding(0, 2, binding2)

    binding3 = Binding("r_sampler", "sampler/filtering", material.map,"FRAGMENT")
    shader.define_binding(0, 3, binding3)

    return [
        {
            "depth_write": False,
            "depth_test": False,
            "render_shader": shader,
            "primitive_topology": "triangle-list",
            "cull_mode": wgpu.CullMode.none,
            "indices": (range(n), range(1)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": [binding, binding1, binding2, binding3],
        },
    ]


canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()
camera = gfx.PerspectiveCamera(45, 640 / 480, 1, 2500)
camera.position.set(0, 400, 400 * 3)


# Lights

scene.add( gfx.AmbientLight( '#222222' ) )
directional_light = gfx.DirectionalLight( 0xffffff, 1 )
directional_light.position.set( 1, 1, 1 ).normalize()
scene.add( directional_light )
point_light = gfx.PointLight( 0xffffff, 2, 800 )
scene.add( point_light )

light_helper = gfx.PointLightHelper(point_light, size=4)
scene.add( light_helper )


env_map_path = Path(__file__).parent / "textures" / "Park2"
env_map_urls = ['posx.jpg', 'negx.jpg',
                    'posy.jpg', 'negy.jpg', 
                    'posz.jpg', 'negz.jpg']


data = []
for env_url in env_map_urls:
    data.append(imageio.imread(env_map_path / env_url, pilmode='RGBA'))

env_data = np.stack(data, axis = 0)

tex_size = env_data.shape[1],  env_data.shape[2], 6

tex = gfx.Texture(env_data, dim=2, size=tex_size)
tex.generate_mipmap = True
env_map = tex.get_view(view_dim="cube", layer_range=range(6), address_mode = "repeat", filter="linear")


cube_width = 400
numbers_per_side = 5
sphere_radius = ( cube_width / numbers_per_side ) * 0.8 * 0.5
step_size = 1.0 / numbers_per_side

geometry = gfx.sphere_geometry( sphere_radius, 32, 16 )

index = 0
alpha = 0.0
while(alpha<=1.0):
    beta = 0.0
    while(beta<=1.0):
        gamma = 0.0
        while(gamma<=1.0):
            material = gfx.MeshStandardMaterial(
                color = hls_to_rgb(alpha, 0.5, gamma * 0.5 + 0.1),
                metalness = beta,
                roughness = 1.0 - alpha
            )

            if index % 2 != 0:
                material.env_map = env_map

            mesh = gfx.Mesh( geometry, material )

            mesh.position.x = alpha * 400 - 200
            mesh.position.y = beta * 400 - 200
            mesh.position.z = gamma * 400 - 200
            scene.add( mesh )
            index += 1

            gamma += step_size
        beta += step_size
    alpha += step_size

        




scene2 = gfx.Scene()
background = Skybox(SkyboxMaterial(map=env_map))
scene2.add(background)

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

def animate():
    timer = time.time() * 0.25
    controller.update_camera(camera)

    point_light.position.x = math.sin( timer * 7 ) * 300
    point_light.position.y = math.cos( timer * 5 ) * 400
    point_light.position.z = math.cos( timer * 3 ) * 300

    renderer.render(scene2, camera, flush = False)
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()



