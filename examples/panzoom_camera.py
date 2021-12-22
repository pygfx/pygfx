"""
Example showing orbit camera controls.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

axes = gfx.AxesHelper(length=250)
scene.add(axes)

background = gfx.Background(None, gfx.BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
scene.add(background)

im = imageio.imread("imageio:astronaut.png")
tex = gfx.Texture(im, dim=2)
geometry = gfx.plane_geometry(512, 512)
material = gfx.MeshBasicMaterial(map=tex.get_view(filter="linear"))
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(0, 0, 500)
controls = gfx.PanZoomControls(camera.position.clone())
controls.apply_default_interaction(canvas, camera)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
