"""
Streamed Image stitching
========================

Show stitching of images using weighted blending. The alpha value of the
images are used as weights.

In contrast to the example in `examples/feature_demo/image_stitching.py`,
this example uses streamed rendering, which means that that the images
are rendered one at a time, each in their own pass.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop

# from rendercanvas.offscreen import RenderCanvas, loop
import pygfx as gfx
import numpy as np

canvas = RenderCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

camera = gfx.OrthographicCamera()
camera.local.scale_y = -1

# Set up camera controller for user interaction
controller = gfx.PanZoomController(camera, register_events=renderer)

# A prior needs to be used to setup the camera correctly since we don't have the full scene information.
# In this case, we copied the parameters from the feature_demo/image_stitching.py example.
camera.set_state(
    {
        "position": np.asarray([411.5, 255.5, 0.0]),
        "rotation": np.asarray([0.0, -0.0, -0.0, 1.0]),
        "scale": np.asarray([1.0, -1.0, 1.0]),
        "reference_up": np.asarray([0.0, 1.0, 0.0]),
        "fov": 0.0,
        "width": 865.2,
        "height": 537.6,
        "depth": 701.4000000000001,
        "zoom": 1.0,
        "maintain_aspect": True,
        "depth_range": None,
    }
)


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

image_names = ["wood.jpg", "bricks.jpg"]

# Create the text and background objects once, outside the animation loop
scene_text_and_background = gfx.Scene()
text = gfx.Text("Streamed image stitching", font_size=64, anchor="top-left")
text.local.scale_y = -1
scene_text_and_background.add(text)
# Add a colorful background
scene_text_and_background.add(gfx.Background.from_color("#C04848", "#480048"))


def animate():
    # Here the images are read in from disk.
    # The PyGFX scene is created
    # And then the memory can be cleared from the GPU and the CPU
    # To make space for the next image
    x = 0
    for image_name in image_names:
        rgb = iio.imread(f"imageio:{image_name}")[:, :, :3]  # Drop alpha if it has it
        rgba = np.empty((*rgb.shape[:2], 4), np.uint8)
        weights = create_pyramid_weights(*rgb.shape[:2])
        weights = (weights * 255).astype("u1")
        rgba = np.dstack([rgb, weights])
        image = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(rgba, dim=2)),
            gfx.ImageBasicMaterial(
                clim=(0, 255),
                blending=blending,
                # We want these images to write to the depth map
                # since the background should be placed behind them.
                depth_write=True,
                # The standard depth test is "<"
                # which won't work for blending these kinds of images
                # since they are positioned at the exact same depth
                depth_compare="<=",
            ),
        )
        scene = gfx.Scene()
        scene.add(image)
        image.local.x = x
        renderer.render(scene, camera, flush=False)
        x += rgba.shape[1] - 200

    # Now that we are done streaming in the images, we can add the overlay text
    # and the background to render the final image.
    # Once we have finished overlaying the text, we can flush the renderer
    # to ensure that the final image is rendered.
    renderer.render(scene_text_and_background, camera)


canvas.request_draw(animate)

if __name__ == "__main__":
    loop.run()
