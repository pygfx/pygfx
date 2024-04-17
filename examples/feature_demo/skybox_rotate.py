"""
Use a Skybox
============

Example with a skybox background in a rotating scene.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

# Read the image
# The order of the images is already correct for GPU cubemap texture sampling
im = iio.imread("imageio:meadow_cube.jpg")

# Turn it into a 3D image (a 4d nd array)
width = height = im.shape[1]
im.shape = -1, width, height, 3

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Create cubemap texture
tex_size = width, height, 6
tex = gfx.Texture(im, dim=2, size=tex_size)

# And the background image with the cube texture
background = gfx.Background(None, gfx.BackgroundSkyboxMaterial(map=tex))
scene.add(background)

axes = gfx.AxesHelper(5)
scene.add(axes)

camera = gfx.PerspectiveCamera(70)

camera.local.position = (0, 4, 20)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    rot = la.quat_from_euler((0.005, 0.01, 0.01))
    scene.local.rotation = la.quat_mul(rot, scene.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
