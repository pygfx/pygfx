"""
Subdivision
===========

Example showing subdivided polyhedrons.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx


scene = gfx.Group()

material = gfx.MeshBasicMaterial(wireframe=True, side="front")
geometries = [gfx.tetrahedron_geometry(subdivisions=i) for i in range(4)]

for i, geometry in enumerate(geometries):
    polyhedron = gfx.Mesh(geometry, material)
    polyhedron.local.position = (6 - i * 3, 0, 0)
    scene.add(polyhedron)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene, up=(0, 0, 1))
