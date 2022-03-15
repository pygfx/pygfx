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

controls = gfx.OrbitControls(camera.position.clone())
controls.add_default_event_handlers(renderer, canvas, camera)

scene = gfx.Scene()

geometry = gfx.box_geometry(40, 40, 40)
material = gfx.MeshPhongMaterial()
selected_material = gfx.MeshPhongMaterial(color="#FF0000")
hovered_material = gfx.MeshPhongMaterial(color="#FFAA00")

state = {
    "selected": None,
    "hovered": None,
}


def select(obj):
    global state
    if state["selected"]:

        def apply_default_mat(obj):
            if isinstance(obj, gfx.Mesh):
                obj.material = material

        state["selected"].traverse(apply_default_mat)
        state["selected"] = None

    state["selected"] = obj
    if state["selected"]:

        def apply_selected_mat(obj):
            if isinstance(obj, gfx.Mesh):
                obj.material = selected_material

        state["selected"].traverse(apply_selected_mat)


def hover(obj):
    global state
    if state["hovered"]:
        if state["hovered"].material is not selected_material:
            state["hovered"].material = material
            state["hovered"] = None
    state["hovered"] = obj
    if state["hovered"] and state["hovered"].material is not selected_material:
        state["hovered"].material = hovered_material


def select_obj(event):
    select(event.target)
    event.target.set_pointer_capture(event.pointer_id)


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
    controls.update_camera(camera)
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
            cube = gfx.Mesh(geometry, material)
            cube.position.x = randint(-200, 200)
            cube.position.y = randint(-200, 200)
            cube.position.z = randint(-200, 200)
            cube.random_rotation = random_rotation()
            cube.add_event_handler(select_obj, "pointer_down")
            cube.add_event_handler(lambda event: hover(event.target), "pointer_move")
            group.add(cube)

    try:
        # Temporary trick to get mouse move events
        # while no button pressed on Qt backends
        canvas.setMouseTracking(True)
        canvas._subwidget.setMouseTracking(True)
    except ValueError:
        pass
    canvas.request_draw(animate)
    run()
