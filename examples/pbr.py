"""
This example shows a complete PBR rendering effect.
The cubemap of skybox is also the environment cubemap of the helmet.
"""

from pathlib import Path

import numpy as np
import imageio
import trimesh
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run


def load_gltf(path):
    def __parse_texture(pil_image):
        if pil_image is None:
            return None
        m = memoryview(pil_image.tobytes())
        m = m.cast(m.format, shape=(pil_image.size[0], pil_image.size[1], 3))
        tex = gfx.Texture(m, dim=2)
        return tex.get_view(address_mode="repeat", filter="linear")

    def __parse_material(pbrmaterial):
        material = gfx.MeshStandardMaterial()
        material.map = __parse_texture(pbrmaterial.baseColorTexture)

        material.emissive = gfx.Color(*pbrmaterial.emissiveFactor)
        material.emissive_map = __parse_texture(pbrmaterial.emissiveTexture)

        metallic_roughness_map = __parse_texture(pbrmaterial.metallicRoughnessTexture)
        material.roughness = pbrmaterial.roughnessFactor or 1.0
        material.metalness = pbrmaterial.metallicFactor or 1.0
        material.roughness_map = metallic_roughness_map
        material.metalness_map = metallic_roughness_map

        material.normal_map = __parse_texture(pbrmaterial.normalTexture)
        material.normal_scale = (1, -1)

        material.ao_map = __parse_texture(pbrmaterial.occlusionTexture)
        material.ao_map_intensity = 1.0

        material.side = "FRONT"
        return material

    def parse_mesh(mesh):
        visual = mesh.visual
        visual.uv = visual.uv * np.array([1, -1]) + np.array([0, 1])  # uv.y = 1 - uv.y
        return gfx.Mesh(
            gfx.trimesh_geometry(mesh),
            __parse_material(visual.material),
        )

    helmet = trimesh.load(path)
    for node_name in helmet.graph.nodes_geometry:
        transform, geometry_name = helmet.graph[node_name]
        current = helmet.geometry[geometry_name]
        current.apply_transform(transform)

    meshes = list(helmet.geometry.values())
    meshes = [parse_mesh(m) for m in meshes]
    return meshes


# Init
canvas = WgpuCanvas(size=(640, 480), title="gfx_pbr")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480, 0.25, 20)
camera.position.set(-1.8, 0.6, 2.7)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = imageio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(env_img, dim=2, size=(cube_size, cube_size, 6))
env_tex.generate_mipmaps = True
env_view = env_tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

# Apply env map to skybox
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_view))
scene.add(background)

# Load meshes, and apply env map
# Note that this lits the helmet already
gltf_path = (
    Path(__file__).parent / "models" / "DamagedHelmet" / "glTF" / "DamagedHelmet.gltf"
)
meshes = load_gltf(gltf_path)
scene.add(*meshes)
m = meshes[0]  # this example has just one mesh
m.material.env_map = env_view

# Add extra light more or less where the sun seems to be in the skybox
scene.add(gfx.SpotLight(color="#444", position=(-500, 1000, -1000)))


def animate():
    controller.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
