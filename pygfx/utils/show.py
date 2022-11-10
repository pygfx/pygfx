"""
Quickly visualize world objects with as
little boilerplate as possible.
"""

import sys

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
    renderer=None,
    controller=None,
    camera=None,
    before_render=None,
    after_render=None,
    draw_function=None
):
    """Display a WorldObject

    This function provides you with the basic scaffolding to visualize a given
    WorldObject in a new window. While it does add scaffolding, it aims to be
    fully customizable so that you can replace each piece as needed.

    Parameters
    ----------
        object : gfx.WorldObject
            The object to show.
        up : gfx.Vector3
            If set, set the default camera controller's up vector to this value.
        canvas : WgpuCanvas
            The canvas to use to display the object.
        renderer : gfx.Renderer
            The renderer to use while drawing the scene.
        controller : gfx.Controller
            The camera controller to use.
        camera : gfx.Camera
            The camera to use.
        before_render : Callable
            A callback that will be executed during each draw call before a new
            render is made.
        after_render : Callable
            A callback that will be executed during each draw call after a new
            render is made.
        draw_function : Callable
            Replaces the draw callback with a custom one. If set both
            `before_render` and `after_render` will have no effect.
        
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
    else:
        event_loop = sys.modules[renderer.target.__module__].run

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
