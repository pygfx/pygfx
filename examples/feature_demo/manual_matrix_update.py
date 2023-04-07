"""
Transform Control without Matrix Updating
=========================================


Example showing transform control flow without matrix auto updating.
"""
# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import imageio.v3 as iio
import pygfx as gfx


group = gfx.Group()

im = iio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2)

material = gfx.MeshBasicMaterial(map=tex)
geometry = gfx.box_geometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.matrix_auto_update = False
    cube.matrix = gfx.linalg.Matrix4().set_position_xyz(350 - i * 100, 0, 0)
    group.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.matrix_auto_update = False
camera.matrix = gfx.linalg.Matrix4().set_position_xyz(0, 0, 500)


def animate():
    for i, cube in enumerate(cubes):
        pos = gfx.linalg.Matrix4().set_position_xyz(350 - i * 100, 0, 0)
        rot = gfx.linalg.Matrix4().extract_rotation(cube.matrix)
        rot.premultiply(
            gfx.linalg.Matrix4().make_rotation_from_euler(
                gfx.linalg.Euler(0.01 * i, 0.02 * i)
            )
        )
        rot.premultiply(pos)
        cube.matrix = rot


if __name__ == "__main__":
    disp = gfx.Display(camera=camera)
    disp.before_render = animate
    disp.show(group)
