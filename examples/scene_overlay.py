"""
Example showing a 3D scene with a 2D overlay.

The idea is to render both scenes, but clear the depth before rendering
the overlay, so that it's always on top.
"""

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Create a canvas and renderer

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(500, 300)))

# Compose a 3D scene

view1 = gfx.View(renderer, gfx.OrthographicCamera(300, 300))

geometry1 = gfx.box_geometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1.0))
cube1 = gfx.Mesh(geometry1, material1)
view1.scene.add(cube1)


# Compose another scene, a 2D overlay

view2 = gfx.View(renderer, gfx.NDCCamera())

positions = np.array(
    [
        [-1, -1, 0.5],
        [-1, +1, 0.5],
        [+1, +1, 0.5],
        [+1, -1, 0.5],
        [-1, -1, 0.5],
        [+1, +1, 0.5],
    ],
    np.float32,
)
geometry2 = gfx.Geometry(positions=positions * 0.9)
material2 = gfx.LineMaterial(thickness=5.0, color=(0.8, 0.0, 0.2, 1.0))
line2 = gfx.Line(geometry2, material2)
view2.scene.add(line2)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube1.rotation.multiply(rot)

    renderer.render_views(view1, view2)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
