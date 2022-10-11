"""
Show an image and print the x, y image data coordinates for click events.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# %% add image

im = imageio.imread("imageio:astronaut.png")

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(256, 256, 0)
camera.scale.y = -1


def event_handler(event):
    print(f"Canvas click coordinates: {event.x, event.y}\n"
          f"Click position in coordinate system of image, i.e. data coordinates of click event: {event.pick_info['index']}\n"
          f"Other `pick_info`: {event.pick_info}")


image.add_event_handler(event_handler, "click")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
