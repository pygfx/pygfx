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
controls.add_default_event_handlers(renderer, canvas, camera)

gizmo = gfx.TransformGizmo(cube)
gizmo.add_default_event_handlers(renderer, camera)


def animate():
    # We render the scene, and then the gizmo on top,
    # as an overlay, so that it's always on top.
    controls.update_camera(camera)
    renderer.render(scene, camera, flush=False)
    renderer.render(gizmo, camera, clear_color=False)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
