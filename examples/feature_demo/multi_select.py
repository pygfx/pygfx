"""
Multi-Object Selection
======================


Example demonstrating multi object selection using mouse events.

Hovering the mouse over a cube will highlight it with a bounding box.
Clicking on a cube will select it. Double-clicking a cube will select
all the items from that group (because the group has a double-click
event handler). Holding shift will add to the selection.
"""
# sphinx_gallery_pygfx_render = True

from functools import partial
from random import randint, random

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400
camera.show_pos(((0, 0, 0)))

controller = gfx.OrbitController(camera, register_events=renderer)

scene = gfx.Scene()
scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

geometry = gfx.box_geometry(40, 40, 40)
default_material = gfx.MeshPhongMaterial()
selected_material = gfx.MeshPhongMaterial(color="#FF0000")
hover_material = gfx.MeshPhongMaterial(color="#FFAA00")

outline = gfx.BoxHelper(thickness=3, color="#fa0")
scene.add(outline)

selected_objects = []


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
    if selected_objects and "Shift" not in event.modifiers:
        while selected_objects:
            ob = selected_objects.pop()
            ob.traverse(partial(set_material, default_material))

    # if the background was clicked, we're done
    if isinstance(obj, gfx.Renderer):
        return

    # set selection (group or mesh)
    selected_objects.append(obj)
    obj.traverse(partial(set_material, selected_material))


def hover(event):
    obj = event.target
    if event.type == "pointer_enter":
        obj.add(outline)
        outline.set_transform_by_object(obj, "local", scale=1.1)
    elif outline.parent:
        outline.parent.remove(outline)


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
