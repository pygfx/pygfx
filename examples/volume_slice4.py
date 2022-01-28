"""
Render slices through a volume, by creating a 3D texture, and viewing
it with a VolumeSliceMaterial. Easy because we can just define the view plane.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

voldata = imageio.volread("imageio:stent.npz").astype("float32")
nslices = voldata.shape[0]
index = nslices // 2

vol = gfx.Volume(
    gfx.Geometry(grid=voldata),
    gfx.VolumeSliceMaterial(clim=(0, 2000), plane=(0, 0, -1, index)),
)
scene.add(vol)

camera = gfx.OrthographicCamera(128, 128)
camera.position.set(64, 64, 128)
camera.scale.y = -1  # in this case we tweak the camera, not the plane


@canvas.add_event_handler("wheel")
def handle_wheel_event(event):
    global index
    index = index + event["dy"] / 90
    index = max(0, min(nslices - 1, index))
    vol.material.plane = 0, 0, -1, index
    canvas.request_draw()


@canvas.add_event_handler("pointer_down")
def handle_pointer_event(event):
    info = renderer.get_pick_info((event["x"], event["y"]))
    if "index" in info:
        print(info["index"], info["voxel_coord"])


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
