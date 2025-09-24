"""
Offscreen Rendering
===================

Example demonstrating off-screen rendering.

This uses wgpu's offscreen canvas to obtain the frames as a numpy array.
Note that one can also render to a ``pygfx.Texture`` and use that texture to
decorate an object in another scene.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import os
import tempfile
import webbrowser

import imageio.v3 as iio
import numpy as np
import pygfx as gfx
import pylinalg as la
from rendercanvas.offscreen import RenderCanvas


# Create offscreen canvas, renderer and scene
canvas = RenderCanvas(size=(640, 480), pixel_ratio=1)
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

im = iio.imread("imageio:astronaut.png").astype("float32") / 255
tex = gfx.Texture(im, dim=2)

geometry = gfx.box_geometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.z = 400

rot = la.quat_from_euler((0.5, 1.0), order="XY")
cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    # The offscreen canvas has a draw method that returns a memoryview.
    # Use this to obtain what you normally see on-screen. You should
    # only use an offscreen canvas for e.g. testing or generating images.
    im1 = np.asarray(canvas.draw())
    print("image from canvas.draw():", im1.shape, im1.dtype)  # (480, 640, 4)

    # The renderer also has a snapshot utility. With this you get a snapshot
    # of the internal state (might be at a higher resolution).
    # The use of the snapshot method may change and be improved.
    im2 = renderer.snapshot()
    print("Image from renderer.snapshot():", im2.shape, im2.dtype)  # (960, 1280, 4)

    filename = os.path.join(tempfile.gettempdir(), "pygfx_offscreen.png")
    iio.imwrite(filename, im1)
    # iio.imwrite(filename, np.clip((im2 * 255), 0, 255).astype(np.uint8))
    webbrowser.open("file://" + filename)
