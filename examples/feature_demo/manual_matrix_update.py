"""
Transform Control without Matrix Updating
=========================================


Example showing transform control flow without matrix auto updating.
"""
# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import imageio.v3 as iio
import pygfx as gfx
import pylinalg as la


group = gfx.Group()

im = iio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2)

material = gfx.MeshBasicMaterial(map=tex)
geometry = gfx.box_geometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.local.matrix = la.matrix_make_translation((350 - i * 100, 0, 0))
    group.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.matrix = la.matrix_make_translation((0, 0, 500))


def animate():
    for i, cube in enumerate(cubes):
        cube.local.position = (350 - i * 100, 0, 0)
        rot = la.quaternion_make_from_euler_angles((0.01 * i, 0.02 * i), order="XY")
        cube.local.rotation = la.quaternion_multiply(rot, cube.local.rotation)


if __name__ == "__main__":
    disp = gfx.Display(camera=camera)
    disp.before_render = animate
    disp.show(group)
