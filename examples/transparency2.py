"""
Example showing transparency using three orthogonal planes.
Press space to toggle the order of the planes.
Press 1-6 to select the blend mode.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

sphere = gfx.Mesh(gfx.sphere_geometry(10), gfx.MeshPhongMaterial())

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.5)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.7)))

plane1.rotation.set_from_axis_angle(gfx.linalg.Vector3(1, 0, 0), 1.571)
plane2.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 1, 0), 1.571)
plane3.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 0, 1), 1.571)

scene.add(plane1, plane2, plane3, sphere)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 70
controls = gfx.OrbitControls(camera.position.clone())
controls.add_default_event_handlers(renderer, camera)


@renderer.add_event_handler("key_down")
def handle_event(event):
    if event.key == " ":
        print("Rotating scene element order")
        scene.add(scene.children[0])
        canvas.request_draw()
    elif event.key in "0123456789":
        m = [
            None,
            "opaque",
            "ordered1",
            "ordered2",
            "weighted",
            "weighted_depth",
            "weighted_plus",
        ]
        mode = m[int(event.key)]
        renderer.blend_mode = mode
        print("Selecting blend_mode", mode)


def animate():
    controls.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    run()
