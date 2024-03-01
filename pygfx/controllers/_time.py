from time import perf_counter

from ..utils.viewport import Viewport


class Time:
    def __init__(self, register_events=None):
        self._t0 = None
        self._delta = 0

        if register_events is not None:
            self.register_events(register_events)

    def register_events(self, viewport_or_renderer):
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport),
            "after_flush",
        )

    def handle_event(self, event, viewport):
        now = perf_counter()
        if self._t0 is not None:
            self._delta = now - self._t0
        self._t0 = now

    @property
    def now(self):
        """Returns the current time in seconds."""
        return perf_counter()

    @property
    def t0(self):
        """Returns the time in seconds at which the previous frame was rendered."""
        return self._t0

    @property
    def delta(self):
        """Returns the time in seconds since the last frame."""
        return self._delta
