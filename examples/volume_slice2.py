"""
Render slices through a volume, by creating a 3D texture, with 2D views.
Simple and relatively fast, but no subslices.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

vol = imageio.volread("imageio:stent.npz").astype("float32") / 2000
nslices = vol.shape[0]
index = nslices // 2

tex_size = tuple(reversed(vol.shape))
tex = gfx.Texture(vol, dim=2, size=tex_size)
view = tex.get_view(filter="linear", view_dim="2d", layer_range=range(index, index + 1))

geometry = gfx.plane_geometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=view)
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(200, 200)


@canvas.add_event_handler("wheel")
def handle_event(event):
    global index
    index = index + int(event["dy"] / 90)
    index = max(0, min(nslices - 1, index))
    view = tex.get_view(
        filter="linear", view_dim="2d", layer_range=range(index, index + 1)
    )
    material.map = view
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
