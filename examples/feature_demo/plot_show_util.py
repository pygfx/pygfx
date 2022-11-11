"""
Use gfx.show
============

Demonstrates show utility
"""

import pygfx as gfx

cube = gfx.Mesh(
    gfx.box_geometry(100, 100, 100),
    gfx.MeshPhongMaterial(color="red"),
)

if __name__ == "__main__":
    gfx.show(cube)
