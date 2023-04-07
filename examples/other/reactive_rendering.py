"""
Reactive rendering loop
================================

This example encapsulates all state that has a visual impact on the drawn frame,
and only renders a new frame when there are changes in that state.

The goal of this rendering strategy is to minimize energy and resource consumption.
"""

import atexit
import pickle
from pathlib import Path

from observ import reactive, watch
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


HERE = Path(__file__).parent
state_file = HERE / "reactive_rendering_state.pkl"


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
viewport = gfx.Viewport.from_viewport_or_renderer(renderer)
scene = gfx.Scene()

colors = ["#336699", "#996633"]
cube = gfx.Mesh(
    gfx.box_geometry(1, 1, 1),
    gfx.MeshPhongMaterial(color=colors[0]),
)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 4

controller = gfx.OrbitController(camera.position.clone(), auto_update=False)

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


visual_state = reactive({
    "camera": {
        "position": None,
        "rotation": None,
        "zoom": None,
    },
    "cube": {
        "color": None,
    },
})


def initialize_state():
    visual_state["cube"]["color"] = cube.material.color.hex
    rotation, position, zoom = controller.get_view()
    visual_state["camera"]["position"] = position.to_array()
    visual_state["camera"]["rotation"] = rotation.to_array()
    visual_state["camera"]["zoom"] = zoom


def process_inputs(event):
    if event.type == "pointer_down":
        # toggle between two colors
        visual_state["cube"]["color"] = colors[0] if visual_state["cube"]["color"] == colors[1] else colors[1]

    controller.handle_event(event, viewport, camera)
    rotation, position, zoom = controller.get_view()
    visual_state["camera"]["position"] = position.to_array()
    visual_state["camera"]["rotation"] = rotation.to_array()
    visual_state["camera"]["zoom"] = zoom


def update_scene():
    camera.position = gfx.linalg.Vector3(*visual_state["camera"]["position"])
    camera.rotation = gfx.linalg.Quaternion(*visual_state["camera"]["rotation"])
    camera.zoom = visual_state["camera"]["zoom"]
    cube.material.color = visual_state["cube"]["color"]


frames = 0


def render_frame():
    global frames
    frames += 1
    print(f"frames: {frames}")
    renderer.render(scene, camera)


def process_state_change():
    update_scene()
    canvas.request_draw()


if __name__ == "__main__":
    # restore state from previous session
    # NOTE: ideally we would persist visual_state
    # and restore visual_state only,
    # but the Controller class unfortunately makes this 
    # impossible without also redundantly tracking its state
    if state_file.exists():
        with state_file.open(mode="rb") as fh:
            state = pickle.load(fh)
            controller.load_state(state=state["controller"])
            rotation, position, zoom = controller.get_view()
            visual_state["camera"]["position"] = position.to_array()
            visual_state["camera"]["rotation"] = rotation.to_array()
            visual_state["camera"]["zoom"] = zoom
            visual_state["cube"]["color"] = state["cube"]
    else:
        initialize_state()

    # persist state at end of session
    def persist_controller_state():
        with state_file.open(mode="wb") as fh:
            pickle.dump({
                "controller": controller.save_state(),
                "cube": visual_state["cube"]["color"],
            }, fh)

    atexit.register(persist_controller_state)

    # inputs trigger state changes
    renderer.add_event_handler(
        process_inputs,
        "pointer_down",
        "pointer_move",
        "pointer_up",
        "wheel",
    )

    # state changes trigger draw calls
    # NOTE: in this example, we keep this simple by
    # always updating the whole scene based on all visual_state
    # but of course you could make this much more optimal by setting up
    # more specific watchers, like so (just one way to do it):
    # watcher = watch(lambda: visual_state, lambda: canvas.request_draw, sync=True, deep=True)
    # watcher = watch(lambda: visual_state["camera"], update_camera, sync=True, deep=True)
    # watcher = watch(lambda: visual_state["cube"], update_cube, sync=True, deep=True)
    # additionally we use sync=True because we have not set up a scheduler and event loop integration
    # because it would complicate the example too much
    watcher = watch(lambda: visual_state, process_state_change, sync=True, deep=True, immediate=True)

    # configure draw calls
    canvas.request_draw(render_frame)

    # start!
    run()
