"""
Example showing a single geometric cube.
"""

import pygfx as gfx

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)


@gfx.Display.before_render
def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(cube)
