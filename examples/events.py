"""
Example showing events and clicking on meshes.

Clicking on a cube will select it. Hovering the mouse over a cube
will show a hover color. Double-clicking a cube will select all
the items from that group (because the group has a double-click
event handler).
"""

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
hovered_material = gfx.MeshPhongMaterial(color="#FFAA00")

selected_obj = None


def select(obj):
    global selected_obj
    if selected_obj:

        def apply_default_mat(obj):
            if isinstance(obj, gfx.Mesh):
                obj.material = default_material

        selected_obj.traverse(apply_default_mat)
        selected_obj = None

    selected_obj = obj
    if selected_obj:

        def apply_selected_mat(obj):
            if isinstance(obj, gfx.Mesh):
                obj.material = selected_material

        selected_obj.traverse(apply_selected_mat)


def hover(event):
    if (
        event.type == gfx.EventType.POINTER_LEAVE
        and event.target.material is not selected_material
    ):
        event.target.material = default_material
    elif (
        event.type == gfx.EventType.POINTER_ENTER
        and event.target.material is not selected_material
    ):
        event.target.material = hovered_material


def select_obj(event):
    select(event.target)


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
    # Build up scene
    for _ in range(4):
        group = gfx.Group()
        group.random_rotation = random_rotation()
        group.add_event_handler(
            lambda event: select(event.current_target), "double_click"
        )
        scene.add(group)

        for _ in range(10):
            cube = gfx.Mesh(geometry, default_material)
            cube.position.x = randint(-200, 200)
            cube.position.y = randint(-200, 200)
            cube.position.z = randint(-200, 200)
            cube.random_rotation = random_rotation()
            cube.add_event_handler(select_obj, "click")
            cube.add_event_handler(hover, "pointer_enter", "pointer_leave")
            group.add(cube)

    canvas.request_draw(animate)
    run()
