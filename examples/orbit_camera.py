"""
Example showing orbit camera controller.

Press 's' to save the state, and
press 'l' to load the last saved state.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

scene.add(gfx.AxesHelper(size=250, arrow_start=0.9, arrow_radius=8))

im = imageio.imread("imageio:chelsea.png")
tex = gfx.Texture(im, dim=2).get_view(filter="linear")

material = gfx.MeshBasicMaterial(map=tex, side="front")
geometry = gfx.box_geometry(100, 100, 100)
cubes = [gfx.Mesh(geometry, material) for i in range(8)]
for i, cube in enumerate(cubes):
    cube.position.set(350 - i * 100, 150, 0)
    scene.add(cube)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.set(0, 0, 500)
controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def on_key_down(event):
    if event.key == "s":
        controller.save_state()
    elif event.key == "l":
        controller.load_state()


renderer.add_event_handler(on_key_down, "key_down")


def animate():
    for i, cube in enumerate(cubes):
        rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0.005 * i, 0.01 * i)
        )
        cube.rotation.multiply(rot)

    controller.update_camera(camera)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
