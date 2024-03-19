"""
Möbius strip Geometry
=====================

Example showing two Möbius strips. The one at the bottom actually has
its ends attached, making it a "non-orientable" manifold. It is rendered
with flat shading to avoid artifacts due to the normals flipping where
the ends meet.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx


ob1 = gfx.Mesh(
    gfx.mobius_strip_geometry(1, stitch=False),
    gfx.MeshPhongMaterial(color=(1, 0.5, 0, 1)),
)
ob1.local.y += 0.5

ob2 = gfx.Mesh(
    gfx.mobius_strip_geometry(1, stitch=True),
    gfx.MeshPhongMaterial(color=(1, 0.5, 0, 1), flat_shading=True),
)
ob2.local.y -= 0.5

scene = gfx.Group()
scene.add(ob1, ob2)


if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene)
