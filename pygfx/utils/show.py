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
    AmbientLight,
    DirectionalLight,
)


def show(
    object: WorldObject,
    up=None,
    *,
    canvas=None,
    event_loop=None,
    renderer=None,
    controller=None,
    camera=None,
    before_render=None,
    after_render=None,
    draw_function=None
):
    """Visualize a given WorldObject in a new window with an interactive camera.

    Parameters:
        object (WorldObject): The object to show.
        up (Vector3): Optional. Configure the up vector for the camera controller.
    """

    if isinstance(object, Scene):
        scene = object
    else:
        scene = Scene()
        scene.add(object)

        background = Background(None, BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
        scene.add(background)
        scene.add(AmbientLight(0.2))

    if not camera:
        camera = PerspectiveCamera(70, 16 / 9)
        camera.add(DirectionalLight(0.8))
        scene.add(camera)

    if canvas is None:
        from wgpu.gui.auto import WgpuCanvas, run

        canvas = WgpuCanvas()
        event_loop = run
    elif event_loop is None:
        # TODO: find the matching run function
        raise ValueError(
            "When providing a canvas you also need to provide the event loop."
        )

    if renderer is None:
        renderer = WgpuRenderer(canvas)

    if controller is None:
        look_at = camera.show_object(object)
        controller = OrbitController(camera.position.clone(), look_at, up=up)
        controller.add_default_event_handlers(renderer, camera)

    if draw_function is None:

        def animate():
            if before_render is not None:
                before_render()

            controller.update_camera(camera)
            renderer.render(scene, camera)

            if after_render is not None:
                after_render()

            renderer.request_draw()

        draw_function = animate

    canvas.request_draw(draw_function)
    event_loop()
