"""
Hello World
===========

In this example, we will have a look the render engine equivalent of a hello
world example: Rendering a 3D Cube on screen.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)
scene.add(cube)

# %%
# Bring it to life.
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight())


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
