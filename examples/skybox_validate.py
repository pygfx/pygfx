import numpy as np
import imageio
import pygfx as gfx
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run

canvas = WgpuCanvas(size=(640, 640))
renderer = gfx.renderers.WgpuRenderer(canvas)

camera = gfx.PerspectiveCamera(90, 1 / 1, 0.25, 20)

env_map_path = Path(__file__).parent / "textures" / "cubemap.jpg"

datas = []

data = imageio.imread(Path(env_map_path), pilmode="RGBA")

h = data.shape[0] // 3
w = data.shape[1] // 4

posx = np.ascontiguousarray(data[1 * h : 2 * h, 2 * w : 3 * w])
datas.append(posx)

negx = np.ascontiguousarray(data[1 * h : 2 * h, 0 * w : 1 * w])
datas.append(negx)

posy = np.ascontiguousarray(data[0 * h : 1 * h, 1 * w : 2 * w])
datas.append(posy)

negy = np.ascontiguousarray(data[2 * h : 3 * h, 1 * w : 2 * w])
datas.append(negy)

posz = np.ascontiguousarray(data[1 * h : 2 * h, 1 * w : 2 * w])
datas.append(posz)

negz = np.ascontiguousarray(data[1 * h : 2 * h, 3 * w : 4 * w])
datas.append(negz)

env_data = np.stack(datas, axis=0)

tex_size = env_data.shape[1], env_data.shape[2], 6

tex = gfx.Texture(env_data, dim=2, size=tex_size)

env_map = tex.get_view(
    view_dim="cube", layer_range=range(6), address_mode="repeat", filter="linear"
)

scene = gfx.Scene()
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=env_map))
scene.add(background)

material = gfx.MeshStandardMaterial(roughness=0.01, metalness=1)
material.side = "Front"
material.env_map = env_map

mesh = gfx.Mesh(
    gfx.sphere_geometry(2, 64, 64),
    material,
)


camera.position.set(0, 0, 5)

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():
    controller.update_camera(camera)

    scene.add(mesh)
    renderer.render(scene, camera)
    renderer.request_draw(animate)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
