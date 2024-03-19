"""
Use gfx.show to show a Scene
============================

Demonstrates show utility for a scene
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
import pygfx as gfx

group = gfx.Group()

im = iio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2)

material = gfx.MeshBasicMaterial(map=tex)
geometry = gfx.box_geometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.local.position = (350 - i * 100, 0, 0)
    group.add(cube)

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(group)
