"""
Image stitching
===============

Show stitching of images using the weighted blend mode. The alpha value of the
images are used as weights.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene1 = gfx.Scene()
scene1.add(gfx.Background.from_color("#fff"))

# add image


def create_texcoords_array(ny, nx):
    texcoord_x = np.linspace(0, 1, nx, dtype="f4")
    texcoord_y = np.linspace(0, 1, ny, dtype="f4")
    return np.stack(np.meshgrid(texcoord_x, texcoord_y), axis=2)


def create_pyramid_weights(ny, nx):
    texcoords = create_texcoords_array(ny, nx)
    center_coords = 1 - np.abs(texcoords * 2 - 1)
    return center_coords.min(axis=2)


# TODO: Maybe this can also be achieved using the texcoords in the shader, using a custom shader, or perhaps custom blending

x = 0
for image_name in ["wood.jpg", "bricks.jpg"]:
    rgb = iio.imread(f"imageio:{image_name}")[:, :, :3]  # Drop alpha if it has it
    rgba = np.empty((*rgb.shape[:2], 4), np.uint8)
    weights = create_pyramid_weights(*rgb.shape[:2])
    weights = (weights * 255).astype("u1")
    rgba = np.dstack([rgb, weights])
    image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(rgba, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255), blending="weighted", depth_write=False),
    )
    scene1.add(image)
    image.local.x = x
    x += rgba.shape[1] - 200

scene2 = gfx.Scene()
text = gfx.Text("Image stitching", font_size=64, anchor="top-left")
text.local.scale_y = -1
scene2.add(text)

camera = gfx.OrthographicCamera()
camera.local.scale_y = -1
camera.show_object(scene1, match_aspect=True, scale=1.05)

controller = gfx.PanZoomController(camera, register_events=renderer)


def animate():
    renderer.render(scene1, camera, flush=False)
    renderer.render(scene2, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
