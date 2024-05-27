"""
Showing CPU usage in the title bar
==================================

Convenient for when investigating performance.

"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import time

import psutil
import pygfx as gfx
import pylinalg as la

# import PySide6  # uncomment to use qt instead of glfw


cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)


p = psutil.Process()
p.next_time = 0


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    if time.time() > p.next_time:
        p.next_time = time.time() + 1
        title = f"{disp.canvas.__class__.__name__} {p.cpu_percent()}%"
        disp.canvas.set_title(title)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.stats = True
    disp.show(cube)
