"""
Example demonstrating off-screen rendering.

This uses wgpu's offscreen canvas to obtain the frames as a numpy array.
Note that one can also render to a ``pygfx.Texture`` and use that texture to
decorate an object in another scene.
"""

import imageio.v3 as iio
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas


# Create offscreen canvas, renderer and scene
canvas = WgpuCanvas(640, 480, 1)
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:astronaut.png").astype("float32") / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.box_geometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.5, 1.0))
cube.rotation.multiply(rot)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":

    # The offscreen canvas has a draw method that returns a numpy array.
    # Use this to obtain what you normally see on-screen. You should
    # only use an offscreen canvas for e.g. testing or generating images.
    im1 = canvas.draw()
    print("image from canvas.draw():", im1.shape)  # (480, 640, 4)

    # The renderer also has a snapshot utility. With this you get a snapshot
    # of the internal state (might be at a higher resolution).
    # The use of the snapshot method may change and be improved.
    im2 = renderer.snapshot()
    print("Image from renderer.snapshot():", im2.shape)  # (960, 1280, 4)

    iio.imwrite("offscreen.png", im1)
