import time

from .. import (
    plane_geometry,
    Mesh,
    Group,
    TextGeometry,
    Text,
    TextMaterial,
    ScreenCoordsCamera,
    MeshBasicMaterial,
)


class Stats(Group):
    """A Stats helper which displays performance statistics such
    as FPS and draw time on the screen.

    Parameters
    ----------
    viewport : Viewport, Renderer
        Required for positioning lines of text in the corner of
        the screen.

    """

    def __init__(
        self,
        viewport,
    ):
        from .. import Viewport  # avoid circular import

        super().__init__()

        self._line_height = 16
        font_size = 12
        foreground = "#0f0"
        background = "#020"
        text_material = TextMaterial(color=foreground)

        quad_geometry = plane_geometry()
        quad_geometry.positions.data[..., :2] += [0.5, -0.5]

        self.bg = Mesh(
            quad_geometry,
            MeshBasicMaterial(
                color=background,
                side="both",
                opacity=0.9,
            ),
        )
        # refactor once text bounding boxes are available
        self.bg.local.scale = (90, self._line_height * 2.1, 1)
        self.stats_text = Text(
            TextGeometry(
                text="",
                screen_space=True,
                font_size=font_size,
                anchor="topleft",
            ),
            text_material,
        )
        self.add(self.bg, self.stats_text)

        self.camera = ScreenCoordsCamera()

        # track screen size for line positioning
        self._viewport = Viewport.from_viewport_or_renderer(viewport)
        self._update_positions()
        self._viewport.renderer.add_event_handler(
            self._update_positions,
            "resize",
        )

        # flag used to skip the first frame
        # which typically has all the CPU->GPU transfer and
        # shader compilation overhead
        self._init = False

        # performance trackers
        self._tmin = 1e10
        self._tmax = 0
        self._tbegin = None
        self._tprev = self._tbegin
        self._frames = 0
        self._fmin = 1e10
        self._fmax = 0
        # Sentinel value of None indicates that the fps has never been computed
        self._fps = None

    def _update_positions(self, event=None):
        _, height = self._viewport.logical_size
        self.stats_text.local.position = (0, height, 0)
        self.bg.local.position = (0, height, 0.1)

    def start(self):
        if not self._init:
            return

        self._tbegin = time.perf_counter_ns()
        if self._tprev is None:
            self._tprev = self._tbegin

    def stop(self):
        if not self._init:
            self._init = True
            return

        t = time.perf_counter_ns()
        self._frames += 1

        delta = round((t - self._tbegin) / 1_000_000)
        self._tmin = min(self._tmin, delta)
        self._tmax = max(self._tmax, delta)

        if t >= self._tprev + 1_000_000_000:
            # update FPS counter whenever a second has passed
            fps = round(self._frames / ((t - self._tprev) / 1_000_000_000))
            self._tprev = t
            self._frames = 0
            self._fmin = min(self._fmin, fps)
            self._fmax = max(self._fmax, fps)
            self._fps = fps

        text = f"{delta} ms ({self._tmin}-{self._tmax})"
        if self._fps is not None:
            text += f"\n{self._fps} fps ({self._fmin}-{self._fmax})"
        self.stats_text.geometry.set_text(text)

    def render(self, flush=True):
        self._viewport.render(self, self.camera, flush=flush)

    def __enter__(self):
        self.start()

    def __exit__(self, *exc):
        self.stop()
