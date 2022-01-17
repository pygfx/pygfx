"""
Example showing subdivided polyhedrons.
"""

import pygfx as gfx


scene = gfx.Scene()

nm = gfx.MeshNormalLinesMaterial()
material = gfx.MeshPhongMaterial(wireframe=True, side="FRONT")
geometries = [
    gfx.tetrahedron_geometry(subdivisions=0),
    gfx.octahedron_geometry(subdivisions=0),
    gfx.icosahedron_geometry(subdivisions=0),
    gfx.dodecahedron_geometry(subdivisions=0),
]

for i, geometry in enumerate(geometries):
    polyhedron = gfx.Mesh(geometry, material)
    npolyhedron = gfx.Mesh(geometry, nm)
    polyhedron.position.set(6 - i * 3, 0, 0)
    npolyhedron.position.set(6 - i * 3, 0, 0)
    scene.add(polyhedron)
    scene.add(npolyhedron)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)


if __name__ == "__main__":
    gfx.show(scene, up=gfx.linalg.Vector3(0, 0, 1))
