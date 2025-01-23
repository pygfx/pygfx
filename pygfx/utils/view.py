import pygfx as gfx


event_types = [
    "resize",
    "pointer_down",
    "pointer_move",
    "pointer_up",
    "key_down",
    "key_up",
    "wheel",
    # "prepare_draw",
    # "animate"
]


class View:
    """The view is a convenience object that combines a renderer, scene, camera.

    The view is a relatively simple object, but significantly reduces boilerplate code.
    You are encouraged to look at the source code to see how it works.
    """

    _canvas = None
    _renderer = None
    _scene = None
    _camera = None
    _controller = None
    _rect = None
    _rect_in_pixels = None

    def __init__(
        self,
        canvas=None,
        *,
        renderer=None,
        rect=None,
        scene=None,
        camera=None,
        controller=None,
        background_color=None,
    ):
        if canvas is None:
            from rendercanvas.auto import RenderCanvas

            canvas = RenderCanvas()
        self.canvas = canvas

        self.renderer = renderer if renderer is not None else gfx.WgpuRenderer(canvas)

        auto_scene = False
        if scene is None:
            auto_scene = True
            scene = gfx.Scene()
            if background_color is None:
                background_color = "#646464", "#a9a9a9"
            elif isinstance(background_color, str):
                background_color = (background_color,)
            elif isinstance(background_color, (list, tuple)):
                if isinstance(background_color[0], (int, float)):
                    background_color = (background_color,)
            background = gfx.Background(None, gfx.BackgroundMaterial(*background_color))
            scene.add(background)
            scene.add(gfx.AmbientLight())
        self.scene = scene

        self._camera_needs_init = not isinstance(camera, gfx.Camera)
        self.camera = camera if camera is not None else "perspective"

        if auto_scene:
            cam_has_light = next(
                self.camera.iter(lambda x: isinstance(x, gfx.Light)), None
            )
            if not cam_has_light:
                self.camera.add(gfx.DirectionalLight())
            self.scene.add(self.camera)

        if controller is None and camera == "2d":
            controller = "2d"
        self.controller = controller if controller is not None else "orbit"
        self.rect = rect if rect is not None else ("0%", "0%", "100%", "100%")

    # --- properties

    @property
    def canvas(self):
        """The canvas that this view renders to.

        Can be a ``rendercanvas.RenderCanvas``, or ``wgpu.gui.WgpuGuiCanvas``.
        """
        return self._canvas

    @canvas.setter
    def canvas(self, canvas):
        # todo: drawing should really be done via an event!
        # Disconnect from the previous canvas
        if self._canvas is not None:
            self._canvas.request_draw(lambda: None)
            self._canvas.remove_event_handler(self._handle_event, *event_types)
        # Connect new canvas
        self._canvas = canvas
        self._canvas.request_draw(self._render)
        self._canvas.add_event_handler(self._handle_event, *event_types)

    @property
    def renderer(self):
        """The renderer to use for this view."""
        return self._renderer

    @renderer.setter
    def renderer(self, renderer):
        if not isinstance(renderer, gfx.Renderer):
            raise TypeError("View renderer must be a gfx.Renderer.")
        self._renderer = renderer

    @property
    def scene(self):
        """The scene that this view renders."""
        return self._scene

    @scene.setter
    def scene(self, scene):
        if not isinstance(scene, gfx.WorldObject):
            raise TypeError("View scene must be a gfx.WorldObject.")
        self._scene = scene

    @property
    def camera(self):
        """The camera used to render this scene."""
        return self._camera

    @camera.setter
    def camera(self, camera):
        if isinstance(camera, str):
            if camera.lower() in ["2d", "ortho", "orthographic"]:
                camera = gfx.OrthographicCamera()
            elif camera.lower() in ["3d", "persp", "perspective"]:
                camera = gfx.PerspectiveCamera()
            else:
                raise TypeError(f"View camera str invalid: '{camera}'")
        elif not isinstance(camera, gfx.Camera):
            raise TypeError("View camera must be a gfx.Camera or str.")
        # Disconnect controller
        if self._camera is not None and self._controller is not None:
            self._controller.remove_camera(self._camera)
        # Connect new camera
        self._camera = camera
        if self._controller is not None:
            self._controller.add_camera(self._camera)

    @property
    def controller(self):
        """The controller used to control the camera."""
        # todo: make it possible to have no controller, or maybe a NullController or somethin?
        return self._controller

    @controller.setter
    def controller(self, controller):
        # Check
        if isinstance(controller, str):
            if controller.lower() in ["2d", "panzoom"]:
                controller = gfx.PanZoomController()
            elif controller.lower() == "orbit":
                controller = gfx.OrbitController()
            elif controller.lower() == "trackball":
                controller = gfx.TrackballController()
            elif controller.lower() == "fly":
                controller = gfx.FlyController()
            else:
                raise TypeError(f"View controller str invalid: '{controller}'")
        elif not isinstance(controller, gfx.Controller):
            raise TypeError("View controller must be a gfx.Controller or str.")
        # Disconnect old controller
        if self._controller is not None:
            self._controller.remove_camera(self._camera)
        # Connect new controller
        self._controller = controller
        self._controller.add_camera(self._camera)

    @property
    def rect(self):
        """The rect (x, y, w, h) for this view.

        Each element is a string e.g. '50% + 4px'. Supported units are 'px' and '%'.
        Supported operarors are '+' and '-'.
        """
        return self._rect

    @rect.setter
    def rect(self, rect):
        if not (isinstance(rect, (tuple, list)) and len(rect) == 4):
            raise TypeError("View rect must be a 4-element tuple.")
        rect2 = []
        for i in rect:
            if isinstance(i, (int, float)):
                rect2.append(f"{i}px")
            elif isinstance(i, str):
                rect_item_to_pixels(i, 100)  # check validity
                rect2.append(i)
            else:
                raise TypeError("View rect elements must be number or str.")
        self._rect = tuple(rect2)

    @property
    def rect_in_pixels(self):
        """The rect (x, y, w, h) expressed in logical pixels."""
        if self._rect_in_pixels is None:
            w, h = self._canvas.get_logical_size()
            refs = [w, h, w, h]
            self._rect_in_pixels = [
                rect_item_to_pixels(i, r) for i, r in zip(self._rect, refs)
            ]
        return self._rect_in_pixels

    # --- methods

    def is_inside(self, x, y):
        """Get whether the given positio (in logical pixels) is inside the viewport rect."""

        vp = self.rect_in_pixels
        return vp[0] <= x <= vp[0] + vp[2] and vp[1] <= y <= vp[1] + vp[3]

    def _handle_event(self, event):
        event_type = event["event_type"]

        # Invalide cached absolute rect
        if event_type == "resize":
            self._rect_in_pixels = None

        # todo: make event positions relative to this view
        # Pass certain events to the controller
        if event_type in [
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "key_down",
            "key_up",
            "wheel",
            "before_render",
        ]:
            if self._controller is not None:
                self._controller.handle_event_dict(event, self)

    def _render(self):
        if self._camera_needs_init:
            self._camera_needs_init = False
            self._camera.show_object(self._scene)

        # todo: use rendercanvas before_draw instead
        event = {"event_type": "before_render"}
        self._controller.handle_event_dict(event, self)

        self._renderer.render(
            self._scene,
            self._camera,
            flush=False,  # rect=self.rect_in_pixels
        )
        self._renderer.flush()
        self._canvas.request_draw()


def rect_item_to_pixels(s, ref):
    """Convert a css-like position/size representation into a number in logical pixels."""
    parts = s.split()
    value = 0
    cur_op = "+"
    for part in parts:
        if part in ("-", "+"):
            cur_op = part
        elif part[0].isnumeric() and part.endswith(("px", "%")):
            part_without_unit = part[:-2] if part.endswith("px") else part[:-1]
            try:
                v = float(part_without_unit)
            except ValueError:
                raise ValueError(
                    f"Cannot interpret {part_without_unit!r} in {s!r}."
                ) from None
            if part.endswith("%"):
                v = (v / 100.0) * ref
            if cur_op == "+":
                value += v
            elif cur_op == "-":
                value -= v
            else:
                assert False, "should not happen"  # noqa
        else:
            raise ValueError(f"Cannot interpret {s!r}, don't know what {part!r} means.")
    return value
