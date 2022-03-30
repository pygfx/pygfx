"""
Example showing a single geometric cube.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
view = gfx.View(renderer)

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)
view.scene.add(cube)

view.camera = gfx.PerspectiveCamera(70, 16 / 9, position=(0, 0, 400))


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render_views(view)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
