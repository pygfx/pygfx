"""
Use a Skybox
============

Example with a skybox background.

Inspired by https://github.com/gfx-rs/wgpu-rs/blob/master/examples/skybox/main.rs
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

# Let's add some cubes to make the scene less boring
cubes = []
for pos in (-600, 0, -600), (-600, 0, +600), (+600, 0, -600), (+600, 0, +600):
    clr = (0.5, 0.6, 0.0, 1.0)
    cube = gfx.Mesh(gfx.box_geometry(200, 200, 200), gfx.MeshPhongMaterial(color=clr))
    cube.local.position = pos
    cubes.append(cube)
    scene.add(cube)


camera = gfx.PerspectiveCamera(70)

light = gfx.DirectionalLight(0.6)
light.local.position = (0, -100, 0)
scene.add(gfx.AmbientLight(0.4), light)


def animate():
    rot = la.quat_from_euler((0.01, 0.02), order="XY")
    for cube in cubes:
        cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    rot = la.quat_from_euler((0, 0.005), order="XY")
    camera.local.rotation = la.quat_mul(rot, camera.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
