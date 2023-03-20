"""
Stats helper (Display)
======================

Demonstrates how to display performance statistics such as FPS
and draw time on screen, using the Display utility.
"""

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import pygfx as gfx

box = gfx.Mesh(
    gfx.box_geometry(5, 5, 5),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)


def animate():
    # Rotate the cube
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    box.rotation.multiply(rot)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.stats = True
    disp.before_render = animate
    disp.show(box)
