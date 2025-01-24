"""
Paint to a texture directly
===========================

This example shows an application that allows the user to paint an image
onto a texture.

The texture does not have a representation in local memory; the GPU
texture is manipilated directly using ``send_data()``.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

scene.add(gfx.Background.from_color("#cde"))

# Create texture to draw to. We set the force_contiguous flag because we will sync the texture regularly
tex = gfx.Texture(
    size=(500, 100, 1),
    dim=2,
    format=wgpu.TextureFormat.r8unorm,
    usage=wgpu.TextureUsage.COPY_DST,
    force_contiguous=True,
)

image = gfx.Image(
    gfx.Geometry(grid=tex),
    gfx.ImageBasicMaterial(clim=(0, 200), pick_write=True),
)
scene.add(image)


camera = gfx.OrthographicCamera()
camera.show_rect(-5, 105, -5, 105)


@image.add_event_handler("pointer_down", "pointer_move")
def pick(event):
    if event.modifiers:
        return
    if 1 in event.buttons or 2 in event.buttons:
        info = event.pick_info
        if "index" in info:
            x, y = info["index"]
            chunk = np.zeros((1, 1), np.uint8)
            if 1 in event.buttons:
                chunk.fill(200)
            tex.send_data((x, y, 0), chunk)
            renderer.request_draw()


controller = gfx.PanZoomController(camera, register_events=renderer)

controller.controls = {
    "shift+mouse1": ("pan", "drag", (1, 0)),
    "wheel": ("pan", "push", (-0.05, 0)),
    "mouse3": ("quickzoom", "peek", 2),
}

if __name__ == "__main__":
    renderer.request_draw(lambda: renderer.render(scene, camera))
    run()
