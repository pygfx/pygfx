"""
Example showing subdivided polyhedrons.
"""

import pygfx as gfx


scene = gfx.Scene()

material = gfx.MeshPhongMaterial(wireframe=True, side="FRONT")
geometries = [
    gfx.tetrahedron_geometry(subdivisions=3),
    gfx.octahedron_geometry(subdivisions=3),
    gfx.icosahedron_geometry(subdivisions=3),
    gfx.dodecahedron_geometry(subdivisions=3),
]

polyhedrons = [gfx.Mesh(g, material) for g in geometries]
for i, polyhedron in enumerate(polyhedrons):
    polyhedron.position.set(6 - i * 3, 0, 0)
    scene.add(polyhedron)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)


if __name__ == "__main__":
    gfx.show(scene, up=gfx.linalg.Vector3(0, 0, 1))
