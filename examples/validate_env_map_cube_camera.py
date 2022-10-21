# test_example = true

import numpy as np
import imageio
import pygfx as gfx
from pathlib import Path
from wgpu.gui.auto import WgpuCanvas, run
from pygfx.linalg import Vector3

# from pygfx.utils.cube_camera import CubeCamera

canvas = WgpuCanvas(size=(800, 600))
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
background = gfx.Skybox(gfx.SkyboxMaterial(map=env_map))
scene.add(background)

# cube_camera = CubeCamera()
# camera_px, camera_nx, camera_py, camera_ny, camera_pz, camera_nz = cube_camera.children

# todo: use CubeCamera instead when it's ready

fov = 90
aspect = 1
near = 0.1
far = 100

camera_px = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_px.up.set(0, 1, 0)
camera_px.look_at(Vector3(-1, 0, 0))

camera_nx = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_nx.up.set(0, 1, 0)
camera_nx.look_at(Vector3(1, 0, 0))

camera_py = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_py.up.set(0, 0, -1)
camera_py.look_at(Vector3(0, 1, 0))

camera_ny = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_ny.up.set(0, 0, 1)
camera_ny.look_at(Vector3(0, -1, 0))

camera_pz = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_pz.up.set(0, 1, 0)
camera_pz.look_at(Vector3(0, 0, 1))

camera_nz = gfx.PerspectiveCamera(fov, aspect, near, far)
camera_nz.up.set(0, 1, 0)
camera_nz.look_at(Vector3(0, 0, -1))


def animate():
    renderer.render(scene, camera_px, rect=(400, 200, 200, 200), flush=False)
    renderer.render(scene, camera_nx, rect=(0, 200, 200, 200), flush=False)
    renderer.render(scene, camera_py, rect=(200, 0, 200, 200), flush=False)
    renderer.render(scene, camera_ny, rect=(200, 400, 200, 200), flush=False)
    renderer.render(scene, camera_pz, rect=(200, 200, 200, 200), flush=False)
    renderer.render(scene, camera_nz, rect=(600, 200, 200, 200))


renderer.request_draw(animate)


if __name__ == "__main__":
    # renderer.request_draw(animate)
    run()
