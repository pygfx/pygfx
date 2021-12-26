"""
Quickly visualize world objects with as
little boilerplate as possible.
"""

from .. import (
    Background,
    BackgroundMaterial,
    WgpuRenderer,
    WorldObject,
    Scene,
    PerspectiveCamera,
    OrbitControls,
)


def show(object: WorldObject, up=None):
    """Visualize a given WorldObject in a new window with an interactive camera.

    Parameters:
        object (WorldObject): The object to show.
        up (Vector3): Optional. Configure the up vector for the camera controls.
    """
    from wgpu.gui.auto import WgpuCanvas, run

    if isinstance(object, Scene):
        scene = object
    else:
        scene = Scene()
        scene.add(object)

        background = Background(None, BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
        scene.add(background)

    camera = PerspectiveCamera(70, 16 / 9)
    look_at = camera.show_object(object)

    canvas = WgpuCanvas()
    renderer = WgpuRenderer(canvas)

    controls = OrbitControls(camera.position.clone(), look_at, up=up)
    controls.add_default_event_handlers(canvas, camera)

    def animate():
        controls.update_camera(camera)
        renderer.render(scene, camera)

    canvas.request_draw(animate)
    run()
