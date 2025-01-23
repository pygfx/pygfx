"""
Sphere Geometry
===============

Example showing different types of geometric cylinders.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas()
view = gfx.View(canvas, background_color="#000")

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=5,
        thickness_space="world",
        major_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -10
view.scene.add(grid)


spheres = [
    (
        (0, 0, -7.5),
        (0, 0.65, 0, 1),
        gfx.sphere_geometry(4.5, phi_length=np.pi * 1.5),
    ),
    ((0, 0, 7.5), (1, 1, 1, 1), gfx.sphere_geometry(4)),
    (
        (15, 0, -7.5),
        (1, 0.3, 0.3, 1),
        gfx.sphere_geometry(4, theta_start=np.pi * 0.25, theta_length=np.pi * 0.50),
    ),
    (
        (15, 0, 7.5),
        (0.35, 0, 0, 1),
        gfx.sphere_geometry(5, width_segments=6),
    ),
    ((-15, 0, -7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(7)),
    ((-15, 0, 7.5), (1, 1, 0.75, 1), gfx.sphere_geometry(5, height_segments=8)),
]
for pos, color, geometry in spheres:
    material = gfx.MeshPhongMaterial(color=color, flat_shading=True)
    wobject = gfx.Mesh(geometry, material)
    wobject.local.position = pos
    view.scene.add(wobject)


if __name__ == "__main__":
    loop.run()
