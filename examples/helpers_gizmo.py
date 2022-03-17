"""
Example to demonstrate the Gizmo that can be used to transform world objects.
Click the center sphere to toggle between object-space, world-space, and screen-space.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(1, 1, 1),
    gfx.MeshPhongMaterial(color="#336699"),
)
scene.add(cube)

scene.add(gfx.GridHelper())

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 4
controls = gfx.OrbitControls(camera.position.clone())

gizmo = gfx.TransformGizmo(cube)


# NOTE: we are working on a new event system, so this event-related
# code will change in the future. At the least we should make these
# `handle_event` methods consistent.
@canvas.add_event_handler("pointer_down", "pointer_move", "pointer_up", "wheel")
def handle_event(event):
    gizmo.handle_event(event, renderer, camera)
    if not event.get("stop_propagation", False):
        controls.handle_event(event, canvas, camera)


def animate():
    # We render the scene, and then the gizmo on top,
    # as an overlay, so that it's always on top.
    controls.update_camera(camera)
    renderer.render(scene, camera, flush=False)
    renderer.render(gizmo, camera, clear_color=False)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
