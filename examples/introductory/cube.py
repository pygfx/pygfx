"""
Hello World
===========

In this example shows how to do the rendering world's hello world: Show a 3D
Cube on screen.

Note: FPS is low since the gallery is rendered on a low-spec CI machine.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s 25fps'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx
import pylinalg as la

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)
cube.local.scale_z = 3


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.stats = True
    disp.show(cube)
