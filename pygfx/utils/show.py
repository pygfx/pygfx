"""
Quickly visualize world objects with as
little boilerplate as possible.
"""

import sys
import numpy as np

from ..objects import (
    Background,
    WorldObject,
    Scene,
    AmbientLight,
    DirectionalLight,
    Light,
)

from ..utils import logger
from ..cameras import Camera, PerspectiveCamera
from ..controllers import OrbitController
from ..materials import BackgroundMaterial
from ..renderers import WgpuRenderer
from ..helpers import Stats


class Display:
    """A Helper to display an object or scene

    This class provides the basic scaffolding needed to visualize a given
    WorldObject. To do so the class chooses a sensible default for each part of
    the full setup unless the value is explicitly set by the user or exists as a
    child of ``object``. For example, it will create a camera unless you
    explicitly set a value for camera or if there is (at least) one camera in
    the scene.

    Parameters
    ----------
    canvas : WgpuCanvas
        The canvas used to display the object. If both ``renderer`` and
        ``canvas`` are set, then the renderer needs to use the set canvas.
    renderer : gfx.Renderer
        The renderer to use while drawing the scene. If both ``renderer``
        and ``canvas`` are set, then the renderer needs to use the set canvas.
    controller : gfx.Controller
        The camera controller to use.
    camera : gfx.Camera
        The camera to use. If not set, Display will use the first camera
        in the scene graph. If there is none, Display will create one.
    before_render : Callable
        A callback that will be executed during each draw call before a new
        render is made.
    after_render : Callable
        A callback that will be executed during each draw call after a new
        render is made.
    draw_function : Callable
        Replaces the draw callback with a custom one. If set, both
        `before_render` and `after_render` will have no effect.
    stats : bool
        Display performance statistics such as FPS and draw time
        in the corner of the screen. Defaults to False.

    """

    def __init__(
        self,
        canvas=None,
        renderer=None,
        controller=None,
        camera=None,
        before_render=None,
        after_render=None,
        draw_function=None,
        stats=False,
    ) -> None:
        self.canvas = canvas
        self.renderer = renderer
        self.controller = controller
        self.camera = camera
        self.scene = None
        self.before_render = before_render
        self.after_render = after_render
        self.draw_function = draw_function or self.default_draw
        self.stats = stats

    def default_draw(self):
        if self.before_render is not None:
            self.before_render()

        flush = True
        if self.stats:
            flush = False
            self.stats.start()

        self.renderer.render(self.scene, self.camera, flush=flush)

        if self.stats:
            self.stats.stop()
            self.stats.render()

        if self.after_render is not None:
            self.after_render()

        self.renderer.request_draw()

    def show(
        self,
        object: WorldObject,
        up=(0, 1, 0),
    ):
        """Display a WorldObject

        This function provides you with the basic scaffolding to visualize a given
        WorldObject in a new window. While it does add scaffolding, it aims to be
        fully customizable so that you can replace each piece as needed.

        Parameters
        ----------
        object : gfx.WorldObject
            The object to show. If it is not a :class:`gfx.Scene <pygfx.Scene>`
            then Display will wrap it into a new scene containing lights and a
            background.
        up : ndarray
            If set, and ``object`` does not contain a controller, set the camera
            controller's up vector to this value.

        Notes
        -----
        If you want to display multiple objects, use :class:`gfx.Group
        <pygfx.Group>` instead of :class:`gfx.Scene <pygfx.Scene>` if you
        want lights and background to be added.

        """

        if self.canvas and self.canvas.is_closed():
            raise RuntimeError(
                "Can not show a closed canvas. Did you repeatedly call `show`?"
            )

        # Process scene

        if isinstance(object, Scene):
            custom_scene = False
            scene = object
        else:
            custom_scene = True
            scene = Scene()
            scene.add(object)

            dark_gray = np.array((169, 167, 168, 255)) / 255
            light_gray = np.array((100, 100, 100, 255)) / 255

            background = Background(None, BackgroundMaterial(light_gray, dark_gray))
            scene.add(background)
            scene.add(AmbientLight())
        self.scene = scene

        if not any(scene.iter(lambda x: isinstance(x, Light))):
            logger.warning(
                "Your scene does not contain any lights. Some objects may not be visible."
            )

        # Process renderer

        if self.renderer is None and self.canvas is None:
            from wgpu.gui.auto import WgpuCanvas

            self.canvas = WgpuCanvas()
            self.renderer = WgpuRenderer(self.canvas)
        elif self.renderer is not None:
            self.canvas = self.renderer.target
        elif self.canvas is not None:
            self.renderer = WgpuRenderer(self.canvas)
        elif self.canvas != self.renderer.target:
            raise ValueError("Display's render target differs from it's canvas.")
        else:
            pass

        # Process stats

        if self.stats is True:
            self.stats = Stats(self.renderer)

        # Process camera

        existing_camera = next(scene.iter(lambda x: isinstance(x, Camera)), None)
        if self.camera:
            pass
        elif existing_camera is not None:
            self.camera = existing_camera
        else:
            self.camera = PerspectiveCamera(70, 4 / 3)
            self.camera.show_object(object, up=up, scale=1.4)
        if custom_scene:
            cam_has_light = next(self.camera.iter(lambda x: isinstance(x, Light)), None)
            if not cam_has_light:
                self.camera.add(DirectionalLight())
            self.scene.add(self.camera)

        # Process controller

        if self.controller is None:
            self.controller = OrbitController(register_events=self.renderer)
        if not self.controller.cameras:
            self.controller.add_camera(self.camera)

        self.canvas.request_draw(self.draw_function)
        sys.modules[self.canvas.__module__].run()


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
    draw_function=None,
):
    """Display a WorldObject

    This function provides you with the basic scaffolding to visualize a given
    WorldObject in a new window. While it does add scaffolding, it aims to be
    fully customizable so that you can replace each piece as needed.

    Parameters
    ----------
    object : gfx.WorldObject
        The object to show. If it is not a :class:`gfx.Scene <pygfx.Scene>`
        then Display will wrap it into a new scene containing lights and a
        background.
    up : ndarray
        If set, and ``object`` does not contain a controller, set the camera
        controller's up vector to this value.
        canvas : WgpuCanvas
        The canvas used to display the object. If both ``renderer`` and
        ``canvas`` are set, then the renderer needs to use the set canvas.
    canvas : WgpuCanvas
        The canvas used to display the object. If both ``renderer`` and
        ``canvas`` are set, then the renderer needs to use the set canvas.
    renderer : gfx.Renderer
        The renderer to use while drawing the scene. If both ``renderer``
        and ``canvas`` are set, then the renderer needs to use the set canvas.
    controller : gfx.Controller
        The camera controller to use.
    camera : gfx.Camera
        The camera to use. If not set, Display will use the first camera
        in the scene graph. If there is none, Display will create one.
    before_render : Callable
        A callback that will be executed during each draw call before a new
        render is made.
    after_render : Callable
        A callback that will be executed during each draw call after a new
        render is made.
    draw_function : Callable
        Replaces the draw callback with a custom one. If set, both
        `before_render` and `after_render` will have no effect.

    Notes
    -----
    If you want to display multiple objects, use :class:`gfx.Group
    <pygfx.Group>` instead of :class:`gfx.Scene <pygfx.Scene>` if you
    want lights and background to be added.

    """

    Display(
        canvas, renderer, controller, camera, before_render, after_render, draw_function
    ).show(object, up)
