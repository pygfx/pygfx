"""
Example to demonstrate the Gizmo that can be used to transform world objects.
Click the center sphere to toggle between object-space, world-space, and screen-space.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
view = gfx.View(renderer)

cube = gfx.Mesh(
    gfx.box_geometry(1, 1, 1),
    gfx.MeshPhongMaterial(color="#336699"),
)
view.scene.add(cube)
view.scene.add(gfx.GridHelper())

view.camera = gfx.PerspectiveCamera(70, 16 / 9, position=(0, 0, 4))
controls = gfx.OrbitControls(view.camera.position.clone())
controls.add_default_event_handlers(view)

gizmo = gfx.TransformGizmo(cube)
gizmo.add_default_event_handlers(view)
gizmo_view = gfx.View(renderer, scene=gizmo, camera=view.camera)


def animate():
    # We render the scene, and then the gizmo on top,
    # as an overlay, so that it's always on top.
    controls.update_camera(view.camera)
    # renderer.render(scene, camera, flush=False)
    # renderer.render(gizmo, camera, clear_color=False)
    renderer.render_views(view, gizmo_view)


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
