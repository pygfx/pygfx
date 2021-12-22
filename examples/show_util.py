"""
Demonstrates show utility
"""

import pygfx as gfx

material = gfx.MeshPhongMaterial()
geometry = gfx.box_geometry(100, 100, 100)
cube = gfx.Mesh(geometry, material)

if __name__ == "__main__":
    gfx.show(cube)
