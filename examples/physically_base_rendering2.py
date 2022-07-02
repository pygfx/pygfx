import numpy as np
import imageio
import trimesh
import wgpu
import pygfx as gfx
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.renderers.wgpu._shadercomposer import Binding, WorldObjectShader


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

    binding3 = Binding("r_sampler", "sampler/filtering", material.map, "FRAGMENT")
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


def load_gltf(path):
    def __parse_texture(pil_image, encoding=None):
        if pil_image is None:
            return None
        # pil_image = pil_image.convert(mode='RGBA')
        data = pil_image.tobytes()
        m = memoryview(data)
        m = m.cast(m.format, shape=(pil_image.size[0], pil_image.size[1], 3))
        tex = gfx.Texture(m, dim=2)
        tex._encoding = encoding
        view = tex.get_view(address_mode="repeat", filter="linear")
        return view

    def __parse_material(pbrmaterial):
        material = gfx.MeshStandardMaterial()
        material.map = __parse_texture(pbrmaterial.baseColorTexture, encoding="srgb")

        material.emissive = gfx.Color(*pbrmaterial.emissiveFactor)
        material.emissive_map = __parse_texture(
            pbrmaterial.emissiveTexture, encoding="srgb"
        )

        metallicRoughnessMap = __parse_texture(pbrmaterial.metallicRoughnessTexture)
        if pbrmaterial.roughnessFactor:
            material.roughness = pbrmaterial.roughnessFactor
        else:
            material.roughness = 1.0
        material.roughness_map = metallicRoughnessMap

        if pbrmaterial.metallicFactor:
            material.metalness = pbrmaterial.metallicFactor
        else:
            material.metalness = 1.0

        material.metalness_map = metallicRoughnessMap

        material.normal_map = __parse_texture(pbrmaterial.normalTexture)
        material.normal_scale = (1, -1)

        material.ao_map = __parse_texture(pbrmaterial.occlusionTexture)
        material.ao_map_intensity = 1.0

        material.side = "FRONT"

        return material

    def parse_mesh(mesh):
        visual = mesh.visual
        visual.uv = visual.uv * np.array([1, -1]) + np.array([0, 1])  # uv.y = 1 - uv.y

        geometry = gfx.trimesh_geometry(mesh)
        material = __parse_material(visual.material)

        return gfx.Mesh(geometry, material)

    helmet = trimesh.load(path)
    for node_name in helmet.graph.nodes_geometry:
        transform, geometry_name = helmet.graph[node_name]
        current = helmet.geometry[geometry_name]
        current.apply_transform(transform)

    meshes = list(helmet.geometry.values())
    meshes = [parse_mesh(m) for m in meshes]
    return meshes


canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()
camera = gfx.PerspectiveCamera(45, 640 / 480, 0.25, 20)

# light = gfx.DirectionalLight(color=(1, 1, 1, 1), intensity=1/ math.pi)
# light.position.set(0, 1, 0)

# scene.add(light)

camera.position.set(-1.8, 0.6, 2.7)

gltf_path = (
    Path(__file__).parent / "models" / "DamagedHelmet" / "glTF" / "DamagedHelmet.gltf"
)
meshes = load_gltf(gltf_path)


env_map_path = Path(__file__).parent / "textures" / "Park2"
env_map_urls = ["posx.jpg", "negx.jpg", "posy.jpg", "negy.jpg", "posz.jpg", "negz.jpg"]


data = []
for env_url in env_map_urls:
    data.append(imageio.imread(env_map_path / env_url, pilmode="RGBA"))

env_data = np.stack(data, axis=0)

tex_size = env_data.shape[1], env_data.shape[2], 6

tex = gfx.Texture(env_data, dim=2, size=tex_size)
tex.generate_mipmap = True
env_map = tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

meshes[0].material.env_map = env_map

scene.add(*meshes)

scene2 = gfx.Scene()
background = Skybox(SkyboxMaterial(map=env_map))

scene2.add(background)

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)
    renderer.render(scene2, camera, flush=False)
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
