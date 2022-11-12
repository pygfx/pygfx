"""
Quickly visualize world objects with as
little boilerplate as possible.
"""

import sys
import numpy as np
import warnings

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
    Light,
    Camera,
)


class RenderCallback:
    """A callback used during the draw call

    This class implements a Descriptor that manages the draw call callbacks of
    Display. It's main purpose is to allow setting the default value of these
    callbacks using a decorator-style syntax::

        @gfx.Display.before_render
        def animate():
            ...

        gfx.show()  # uses animate

    Parameters
    ----------
    default : Callable
        The default callback to use.

    Notes
    -----
    Using the decorator feature above sets the class-level default for the
    callback and affects all instances of Display that use the default value.

    """

    def __init__(self, default=None) -> None:
        self.default_callback = default

    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, instance, type=None):
        # allow decorator-style setting of default
        # callback
        if instance is None:
            return self

        try:
            value = getattr(instance, self.private_name)
        except AttributeError:
            # persist the current class-level default on the instance
            self.__set__(instance, self.default_callback)

            value = self.default_callback

        return value

    def __set__(self, instance, value) -> None:
        setattr(instance, self.private_name, value)

    def __call__(self, callback_fn) -> None:
        self.default_callback = callback_fn


class Display:
    """A Helper to display an object or scene

    This class provides the basic scaffolding needed to visualize a given
    WorldObject. To do so the class chooses a sensible default for each part of
    the full setup (@almarklein: what is the technical term for the full setup?)
    unless the value is explicitly set by the user or exists as a child of
    ``object``. For example, it will create a camera unless you explicitly set a
    value for camera or if there is (at least) one camera in the scene.

    Parameters
    ----------
    canvas : WgpuCanvas
        The canvas used to display the object. If both ``renderer`` and
        ``canvas`` are set, then the renderer needs to use the set canvas.
    renderer : gfx.Renderer
        The renderer to use while drawing the scene. If both ``renderer``
        and ``canvas`` are set, then the renderer needs to use the set canvas.
    controller : gfx.Controller
        The camera controller to use. If not set, Display will use the first
        controller defined on ``object``. If there is none, Display will
        create one.
    camera : gfx.Camera
        The camera to use. If not set, Display will use the first camera
        defined on ``object``. If there is none, Display will create one.
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

    before_render = RenderCallback()
    after_render = RenderCallback()
    draw_function = RenderCallback()

    def __init__(
        self,
        canvas=None,
        renderer=None,
        controller=None,
        camera=None,
        before_render=None,
        after_render=None,
        draw_function=None,
    ) -> None:
        self.canvas = canvas
        self.renderer = renderer
        self.controller = controller
        self.camera = camera
        self.draw_function = draw_function or self.default_draw
        self.scene = None

        if before_render is not None:
            self.before_render = before_render

        if after_render is not None:
            self.after_render = after_render

    def default_draw(self):
        if self.before_render is not None:
            self.before_render()

        self.controller.update_camera(self.camera)
        self.renderer.render(self.scene, self.camera)

        if self.after_render is not None:
            self.after_render()

        self.renderer.request_draw()

    def show(
        self,
        object: WorldObject,
        up=None,
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
        up : gfx.Vector3
            If set, and ``object`` does not contain a controller, set the camera
            controller's up vector to this value.

        Notes
        -----
        If you want to display multiple objects prefer :class:`gfx.Group
        <pygfx.Group>` over :class:`gfx.Scene <pygfx.Scene>` to avoid the pitfall of
        forgetting to add lights to the scene.

        """

        if isinstance(object, Scene):
            scene = object
        else:
            scene = Scene()
            scene.add(object)

            dark_gray = np.array((169, 167, 168, 255)) / 255
            light_gray = np.array((100, 100, 100, 255)) / 255

            background = Background(None, BackgroundMaterial(light_gray, dark_gray))
            scene.add(background)
            scene.add(AmbientLight(), DirectionalLight())
        self.scene = scene

        if not any(self.find_children(Light)):
            warnings.warn(
                "Your scene does not contain any lights. Some objects may not be visible"
            )

        existing_cameras = self.find_children(Camera)
        if self.camera:
            pass
        elif any(existing_cameras):
            self.camera = existing_cameras[0]
        elif not self.camera:
            self.camera = PerspectiveCamera(70, 16 / 9)
            self.camera.add(DirectionalLight())
            self.scene.add(self.camera)

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

        if self.controller is None:
            look_at = self.camera.show_object(object)
            self.controller = OrbitController(
                self.camera.position.clone(), look_at, up=up
            )
            self.controller.add_default_event_handlers(self.renderer, self.camera)

        self.canvas.request_draw(self.draw_function)
        sys.modules[self.canvas.__module__].run()

    # this should probably live on WorldObject
    def find_children(self, clazz):
        """Return all children of the given type"""
        objects = list()

        # can we make traverse a generator?
        # [x for x in self.scene.traverse(filter=lambda x: isinstance(x, clazz))
        self.scene.traverse(
            lambda x: objects.append(x) if isinstance(x, clazz) else None
        )

        return objects


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
    up : gfx.Vector3
        If set, and ``object`` does not contain a controller, set the camera
        controller's up vector to this value.
    canvas : WgpuCanvas
        The canvas used to display the object. If both ``renderer`` and
        ``canvas`` are set, then the renderer needs to use the set canvas.
    renderer : gfx.Renderer
        The renderer to use while drawing the scene. If both ``renderer``
        and ``canvas`` are set, then the renderer needs to use the set canvas.
    controller : gfx.Controller
        The camera controller to use. If not set, Display will use the first
        controller defined on ``object``. If there is none, Display will
        create one.
    camera : gfx.Camera
        The camera to use. If not set, Display will use the first camera
        defined on ``object``. If there is none, Display will create one.
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

    Display(
        canvas, renderer, controller, camera, before_render, after_render, draw_function
    ).show(object, up)
