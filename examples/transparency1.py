"""
Example showing transparency using three overlapping planes.
Press space to toggle the order of the planes.
Press 1-6 to select the blend mode.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.4)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.4)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.4)))

plane1.position.set(-10, -10, 1)
plane2.position.set(0, 0, 2)
plane3.position.set(10, 10, 3)

scene.add(plane1, plane2, plane3)

camera = gfx.OrthographicCamera(100, 100)


@canvas.add_event_handler("key_down")
def handle_event(event):
    if event["key"] == " ":
        print("Rotating scene element order")
        scene.add(scene.children[0])
        canvas.request_draw()
    elif event["key"] in "0123456789":
        m = [
            None,
            "opaque",
            "ordered1",
            "ordered2",
            "weighted",
            "weighted_depth",
            "weighted_plus",
        ]
        mode = m[int(event["key"])]
        renderer.blend_mode = mode
        print("Selecting blend_mode", mode)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
