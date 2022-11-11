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
)


class DecorSet:
    """Set function attributes using decorators.

    This class implements a Descriptor that allows setting
    it's

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

        value = getattr(instance, self.private_name, None)
        if value is not None:
            return value

        # persist callback to prevent it from changing
        self.__set__(instance, self.default_callback)
        return self.default_callback

    def __set__(self, instance, value) -> None:
        setattr(instance, self.private_name, value)

    def __call__(self, callback_fn) -> None:
        self.default_callback = callback_fn


class Display:
    """A Helper to Display Objects

    This class provides the basic scaffolding needed to visualize a given
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
            The renderer to use while drawing the scene. If set, `canvas` is
            ignored.
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

    before_render = DecorSet()
    after_render = DecorSet()
    draw_function = DecorSet()

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
                The object to show.
            up : gfx.Vector3
                If set, set the default camera controller's up vector to this value.
            canvas : WgpuCanvas
                The canvas to use to display the object.
            renderer : gfx.Renderer
                The renderer to use while drawing the scene. If set, `canvas` is
                ignored.
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

        if (
            self.canvas is not None
            and self.renderer is not None
            and self.canvas != self.renderer.target
        ):
            raise ValueError("Display's render target differs from it's canvas.")

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

        if len(self.find_children(Light)) == 0:
            warnings.warn(
                "Your scene does not contain any lights. Some objects may not be visible"
            )

        if not self.camera:
            self.camera = PerspectiveCamera(70, 16 / 9)
            self.camera.add(DirectionalLight())
            self.scene.add(self.camera)

        if self.renderer is not None:
            self.canvas = self.renderer.target
        elif self.canvas is None:
            from wgpu.gui.auto import WgpuCanvas

            self.canvas = WgpuCanvas()
            self.renderer = WgpuRenderer(self.canvas)
        else:
            self.renderer = WgpuRenderer(self.canvas)

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
            The object to show.
        up : gfx.Vector3
            If set, set the default camera controller's up vector to this value.
        canvas : WgpuCanvas
            The canvas to use to display the object.
        renderer : gfx.Renderer
            The renderer to use while drawing the scene. If set, `canvas` is
            ignored.
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

    Display(
        canvas, renderer, controller, camera, before_render, after_render, draw_function
    ).show(object, up)
