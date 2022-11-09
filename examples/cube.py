"""
Example showing a single geometric cube.
"""

import pygfx as gfx


scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight())


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)


if __name__ == "__main__":
    gfx.show(scene, camera=camera, before_render=animate)
