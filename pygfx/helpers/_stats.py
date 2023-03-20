import time

from .. import Group, TextGeometry, Text, TextMaterial, ScreenCoordsCamera


class Stats(Group):
    """A Stats helper that shows a text overlay with performance stats.

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

        self.ms = Text(
            TextGeometry(
                text="",
                screen_space=True,
                font_size=font_size,
                anchor="topleft",
            ),
            TextMaterial(color="#0f4"),
        )
        self.fps = Text(
            TextGeometry(
                text="",
                screen_space=True,
                font_size=font_size,
                anchor="topleft",
            ),
            TextMaterial(color="#0f4"),
        )
        self.add(self.ms, self.fps)

        self.camera = ScreenCoordsCamera()

        # track screen size for line positioning
        self._viewport = Viewport.from_viewport_or_renderer(viewport)
        self._update_positions()
        self._viewport.renderer.add_event_handler(
            self._update_positions,
            "resize",
        )

        # track timings
        self._tmin = 1e10
        self._tmax = 0
        self._tbegin = time.perf_counter_ns()
        self._tprev = self._tbegin
        self._frames = 0
        self._fmin = 1e10
        self._fmax = 0

    def _update_positions(self, event=None):
        height = self._viewport.logical_size[1]
        self.ms.position.set(0, height, 0)
        self.fps.position.set(0, height - self._line_height, 0)

    def start(self):
        self._tbegin = time.perf_counter_ns()

    def stop(self):
        t = time.perf_counter_ns()
        self._frames += 1

        delta = round((t - self._tbegin) / 1_000_000)
        self._tmin = min(self._tmin, delta)
        self._tmax = max(self._tmax, delta)
        self.ms.geometry.set_text(f"{delta} MS ({self._tmin}-{self._tmax})")

        if t >= self._tprev + 1_000_000_000:
            # update FPS counter whenever a second has passed
            fps = round(self._frames / ((t - self._tprev) / 1_000_000_000))
            self._tprev = t
            self._frames = 0
            self._fmin = min(self._fmin, fps)
            self._fmax = max(self._fmax, fps)
            self.fps.geometry.set_text(f"{fps} FPS ({self._fmin}-{self._fmax})")

    def render(self, flush=True):
        self._viewport.render(self, self.camera, flush=flush)

    def __enter__(self):
        self.start()

    def __exit__(self, *exc):
        self.stop()
