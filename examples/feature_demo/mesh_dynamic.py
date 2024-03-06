"""
Mesh dynamic
============

Example showing a Torus knot, dynamically changing what faces are shown.
"""

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import imageio.v3 as iio
import pygfx as gfx


im = iio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2)

obj = gfx.Mesh(gfx.torus_knot_geometry(1, 0.3, 128, 16), gfx.MeshPhongMaterial(map=tex))
obj.geometry.texcoords.data[:, 0] *= 10  # stretch the texture


def animate():
    indices = obj.geometry.indices
    offset = indices.draw_range[0] + 32
    if offset + 640 >= indices.nitems:
        offset = 0
    indices.draw_range = offset, 640


if __name__ == "__main__":
    disp = gfx.Display(before_render=animate, stats=True)
    disp.show(obj)
