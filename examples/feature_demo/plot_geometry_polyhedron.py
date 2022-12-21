"""
Rotating Polyhedra
==================

Example showing multiple rotating polyhedrons.
"""
# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import pygfx as gfx

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
    polyhedron.position.set(6 - i * 3, 0, 0)
    group.add(polyhedron)


def animate():
    for polyhedron in polyhedrons:
        rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.01, 0.02))
        polyhedron.rotation.multiply(rot)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.before_render = animate
    disp.show(group)
