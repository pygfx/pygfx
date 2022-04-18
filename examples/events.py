"""
Example showing events and clicking on meshes.

Clicking on a cube will select it. Hovering the mouse over a cube
will show a hover color. Double-clicking a cube will select all
the items from that group (because the group has a double-click
event handler).
"""

from functools import partial
from random import randint, random

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)

scene = gfx.Scene()

geometry = gfx.box_geometry(40, 40, 40)
default_material = gfx.MeshPhongMaterial()
selected_material = gfx.MeshPhongMaterial(color="#FF0000")
hover_material = gfx.MeshPhongMaterial(color="#FFAA00")

selected_obj = None


def set_material(material, obj):
    if isinstance(obj, gfx.Mesh):
        obj.material = material


def select(event):
    # when this event handler is invoked on non-leaf nodes of the
    # scene graph, event.target will still point to the leaf node that
    # originally triggered the event, so we use event.current_target
    # to get a reference to the node that is currently handling
    # the event, which can be a Mesh, a Group or None (the event root)
    obj = event.current_target

    # prevent propagation so we handle this event only at one
    # level in the hierarchy
    event.stop_propagation()

    # clear selection
    global selected_obj
    if selected_obj:
        selected_obj.traverse(partial(set_material, default_material))
        selected_obj = None

    # if the background was clicked, we're done
    if isinstance(obj, gfx.Renderer):
        return

    # set selection (group or mesh)
    selected_obj = obj
    selected_obj.traverse(partial(set_material, selected_material))


def hover(event):
    obj = event.target

    if obj.material is selected_material:
        return

    obj.material = {
        "pointer_leave": default_material,
        "pointer_enter": hover_material,
    }[event.type]


def random_rotation():
    return gfx.linalg.Quaternion().set_from_euler(
        gfx.linalg.Euler(
            (random() - 0.5) / 100, (random() - 0.5) / 100, (random() - 0.5) / 100
        )
    )


def animate():
    def random_rot(obj):
        if hasattr(obj, "random_rotation"):
            obj.rotation.multiply(obj.random_rotation)

    scene.traverse(random_rot)
    controller.update_camera(camera)
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    renderer.add_event_handler(select, "click")

    # Build up scene
    for _ in range(4):
        group = gfx.Group()
        group.random_rotation = random_rotation()
        group.add_event_handler(select, "double_click")
        scene.add(group)

        for _ in range(10):
            cube = gfx.Mesh(geometry, default_material)
            cube.position.x = randint(-200, 200)
            cube.position.y = randint(-200, 200)
            cube.position.z = randint(-200, 200)
            cube.random_rotation = random_rotation()
            cube.add_event_handler(select, "click")
            cube.add_event_handler(hover, "pointer_enter", "pointer_leave")
            group.add(cube)

    canvas.request_draw(animate)
    run()
