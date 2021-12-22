"""
Show an image displayed the correct way. For historic reasons, image
data (usually) has the first rows representing the top of the image.
Most our camera's default orientations place the origin at the bottom
left.

* We flip the y-axis of either the data or the camera.
* The image should look correct.
* The rocket should be at the right.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


im = imageio.imread("imageio:astronaut.png")
tex = gfx.Texture(im, dim=2)

geometry = gfx.plane_geometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"))

plane = gfx.Mesh(geometry, material)
plane.position = gfx.linalg.Vector3(256, 256, 0)  # put corner at 0, 0
scene.add(plane)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(256, 256, 0)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
