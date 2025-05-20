"""
Image stitching
===============

Show stitching of images using weighted blending. The alpha value of the
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
scene1.add(gfx.Background.from_color("#000"))

# add image


def create_texcoords_array(ny, nx):
    texcoord_x = np.linspace(0, 1, nx, dtype="f4")
    texcoord_y = np.linspace(0, 1, ny, dtype="f4")
    return np.stack(np.meshgrid(texcoord_x, texcoord_y), axis=2)


def create_pyramid_weights(ny, nx):
    texcoords = create_texcoords_array(ny, nx)
    center_coords = 1 - np.abs(texcoords * 2 - 1)
    return center_coords.min(axis=2)


# Define the blending using a dict. We use weighted blending, using the alpha
# channel as weights, and setting the final alpha to 1.
#
# The commented line shows how we could use the shader texcoord to create the
# same effect. This avoids having to create the pyramid alpha channel for the
# image, but it's a less portable solution because it assumes that the shader
# has a 'texcoord' on its varying.
blending = {
    "mode": "weighted",
    "weight": "alpha",
    # "weight": "1.0 - 2.0*max(abs(varyings.texcoord.x - 0.5), abs(varyings.texcoord.y - 0.5))",
    "alpha": "1.0",
}

x = 0
for image_name in ["wood.jpg", "bricks.jpg"]:
    rgb = iio.imread(f"imageio:{image_name}")[:, :, :3]  # Drop alpha if it has it
    rgba = np.empty((*rgb.shape[:2], 4), np.uint8)
    weights = create_pyramid_weights(*rgb.shape[:2])
    weights = (weights * 255).astype("u1")
    rgba = np.dstack([rgb, weights])
    image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(rgba, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255), blending=blending, depth_write=False),
    )
    scene1.add(image)
    image.local.x = x
    x += rgba.shape[1] - 200

scene2 = gfx.Scene()
text = gfx.Text("Image stitching", font_size=64, anchor="top-left")
text.render_order = 1  # render the text on top
text.local.scale_y = -1
scene1.add(text)

camera = gfx.OrthographicCamera()
camera.local.scale_y = -1
camera.show_object(scene1, match_aspect=True, scale=1.05)

controller = gfx.PanZoomController(camera, register_events=renderer)


def animate():
    renderer.render(scene1, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
