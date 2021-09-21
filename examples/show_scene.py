"""
Demonstrates show utility
"""

import imageio
import pygfx as gfx

scene = gfx.Scene()

im = imageio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2).get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, clim=(0, 255))
geometry = gfx.BoxGeometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 0, 0)
    scene.add(cube)

background = gfx.Background(gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)


def animate():
    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.01 * i, 0.02 * i)
        )
        cube.rotation.multiply(rot)


if __name__ == "__main__":
    gfx.show(scene, animate)
