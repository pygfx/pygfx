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
    OrbitController,
    Light,
    AmbientLight,
    DirectionalLight,
)


def show(object: WorldObject, up=None):
    """Visualize a given WorldObject in a new window with an interactive camera.

    Parameters:
        object (WorldObject): The object to show.
        up (Vector3): Optional. Configure the up vector for the camera controller.
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
    scene.add(camera)

    # Add lights if there are none
    light_count = 0

    def check_light(ob):
        nonlocal light_count
        if isinstance(ob, Light):
            light_count += 1

    scene.traverse(check_light, False)
    if not light_count:
        camera.add(DirectionalLight(0.8))
        scene.add(AmbientLight(0.2))

    canvas = WgpuCanvas()
    renderer = WgpuRenderer(canvas)

    controller = OrbitController(camera.position.clone(), look_at, up=up)
    controller.add_default_event_handlers(renderer, camera)

    def animate():
        controller.update_camera(camera)
        renderer.render(scene, camera)

    canvas.request_draw(animate)
    run()
