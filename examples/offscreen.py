"""
Example demonstrating off-screen rendering.
"""

import imageio
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas


# Create offscreen canvas, renderer and scene
canvas = WgpuCanvas(640, 480, 1)
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = imageio.imread("imageio:astronaut.png").astype("float32") / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.5, 1.0))
cube.rotation.multiply(rot)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    # Invoke a draw and get what we'd normally see on screen as a numpy array
    im1 = canvas.draw()
    print(im1.shape)  # (480, 640, 4)
    # Use the renderer's to get intermediate results (internal render targets)
    im2 = renderer.snapshot()
    print(im2.shape)  # (960, 1280, 4)
    imageio.imsave("offscreen.png", im1)
