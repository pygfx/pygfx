"""
Use a Skybox
============

Example with a skybox background in a rotating scene.

"""
# sphinx_gallery_pygfx_render = True

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

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

camera.position.set(0, 4, 20)

controller = gfx.OrbitController(camera, register_events=renderer)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01, 0.01))
    scene.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
