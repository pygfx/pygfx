import time


class Clock:
    """A simple clock for keeping track of time"""

    def __init__(self, auto_start=True):
        self._auto_start = auto_start

        self._start_time = None
        self._last_time = None
        self._elapsed_time = 0
        self._running = False

    def start(self):
        self._start_time = self._last_time = time.perf_counter()
        self._elapsed_time = 0
        self._running = True

    def stop(self):
        self._running = False

    def get_elapsed_time(self):
        self.get_delta()
        return self._elapsed_time

    def get_delta(self):
        diff = 0

        if self._auto_start and not self._running:
            self.start()
            return 0

        if self._running:
            now = time.perf_counter()
            diff = now - self._last_time
            self._last_time = now

            self._elapsed_time += diff

        return diff
