"""
Use gfx.show
============

Demonstrates show utility
"""

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import pygfx as gfx

cube = gfx.Mesh(
    gfx.box_geometry(100, 100, 100),
    gfx.MeshPhongMaterial(color="red"),
)

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(cube)
