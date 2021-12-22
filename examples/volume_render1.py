"""
Render a volume. Shift-click to draw white blobs inside the volume.
"""

import imageio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

voldata = imageio.volread("imageio:stent.npz").astype(np.float32)


tex = gfx.Texture(voldata, dim=3)
vol = gfx.Volume(gfx.Geometry(grid=tex), gfx.VolumeRayMaterial(clim=(0, 2000)))
slice = gfx.Volume(
    gfx.Geometry(grid=tex), gfx.VolumeSliceMaterial(clim=(0, 2000), plane=(0, 0, 1, 0))
)
scene.add(vol, slice)

for ob in (slice, vol):
    ob.position.set(*(-0.5 * i for i in voldata.shape[::-1]))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 500
controls = gfx.OrbitControls(camera.position.clone(), up=gfx.linalg.Vector3(0, 0, 1))
controls.rotate(-0.5, -0.5)


@canvas.add_event_handler("pointer_down", "pointer_up", "pointer_move", "wheel")
def handle_event(event):
    if event["event_type"] == "pointer_down" and "Shift" in event["modifiers"]:
        info = renderer.get_pick_info((event["x"], event["y"]))
        if "voxel_index" in info:
            x, y, z = (max(1, int(i)) for i in info["voxel_index"])
            print("Picking", x, y, z)
            tex.data[z - 1 : z + 1, y - 1 : y + 1, x - 1 : x + 1] = 2000
            tex.update_range((x - 1, y - 1, z - 1), (3, 3, 3))
    else:
        controls.handle_event(event, canvas, camera)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
