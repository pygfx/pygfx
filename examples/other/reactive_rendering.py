"""
Reactive rendering loop
================================

This example encapsulates all state that has a visual impact on the drawn frame,
and only renders a new frame when there are changes in that state.

The goal of this rendering strategy is to minimize energy and resource consumption.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

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
camera.show_object(cube, scale=2)

# Create a controller. Set auto_update to False, because we take control of updates.
controller = gfx.OrbitController(camera, auto_update=False)

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


visual_state = reactive(
    {
        "camera": {k: None for k in camera.get_state().keys()},
        "cube": {
            "color": None,
        },
    }
)


def initialize_state():
    visual_state["cube"]["color"] = cube.material.color.hex
    visual_state["camera"].update(camera.get_state())


def process_inputs(event):
    if event.type == "pointer_down":
        # toggle between two colors
        visual_state["cube"]["color"] = (
            colors[0] if visual_state["cube"]["color"] == colors[1] else colors[1]
        )

    if event.type == "before_render":
        # Let the controller animate, and update our state if it had any
        # actions in progress. One way or another, this code needs to run
        # periodically, because the controller changes state even without
        # events because of inertia.
        camera_state = controller.tick()
        if camera_state:
            visual_state["camera"].update(camera_state)
    else:
        controller.handle_event(event, viewport)


def update_scene():
    camera.set_state(visual_state["camera"])
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
    if state_file.exists():
        with state_file.open(mode="rb") as fh:
            state = pickle.load(fh)
            visual_state["camera"].update(state["camera"])
            visual_state["cube"].update(state["cube"])
    else:
        initialize_state()

    # persist state at end of session
    def persist_scene_state():
        with state_file.open(mode="wb") as fh:
            pickle.dump(
                {
                    "camera": visual_state["camera"],
                    "cube": visual_state["cube"],
                },
                fh,
            )

    atexit.register(persist_scene_state)

    # inputs trigger state changes
    renderer.add_event_handler(
        process_inputs,
        "pointer_down",
        "pointer_move",
        "pointer_up",
        "wheel",
        "before_render",
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
    watcher = watch(
        lambda: visual_state, process_state_change, sync=True, deep=True, immediate=True
    )

    # configure draw calls
    canvas.request_draw(render_frame)

    # start!
    run()
