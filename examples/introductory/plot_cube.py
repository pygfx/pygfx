"""
Hello World
===========

In this example shows how to do the rendering world's hello world: Show a 3D
Cube on screen.
"""
# sphinx_gallery_pygfx_animate = True
# sphinx_gallery_pygfx_target_name = "disp"

import pygfx as gfx

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.show(cube)
