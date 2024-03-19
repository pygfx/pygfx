"""
Rotating Polyhedra
==================

Example showing multiple rotating polyhedrons.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx
import pylinalg as la

group = gfx.Group()

material = gfx.MeshPhongMaterial()
geometries = [
    gfx.tetrahedron_geometry(),
    gfx.octahedron_geometry(),
    gfx.icosahedron_geometry(),
    gfx.dodecahedron_geometry(),
]

polyhedrons = [gfx.Mesh(g, material) for g in geometries]
for i, polyhedron in enumerate(polyhedrons):
    polyhedron.local.position = (6 - i * 3, 0, 0)
    group.add(polyhedron)


def animate():
    for polyhedron in polyhedrons:
        rot = la.quat_from_euler((0.01, 0.02), order="XY")
        polyhedron.local.rotation = la.quat_mul(rot, polyhedron.local.rotation)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.show(group)
