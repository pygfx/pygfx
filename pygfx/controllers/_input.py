from ..utils.viewport import Viewport


class Input:
    def __init__(self, register_events=None):
        # TODO:
        # - controllers
        # - virtual input state (e.g. buttons and axes, so users can remap controls)

        self._key_down = {}
        self._key_up = {}
        self._key_state = {}

        self._pointer_down = {}
        self._pointer_up = {}
        self._pointer_state = {}
        self._pointer_pos = {}
        self._pointer_pos_prev = {}
        self._pointer_delta = {}

        self._wheel_delta = {}

        if register_events is not None:
            self.register_events(register_events)

    def register_events(self, viewport_or_renderer):
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport),
            "pointer_down",
            "pointer_move",
            "pointer_up",
            "wheel",
            "key_down",
            "key_up",
            "after_flush",
        )

    def handle_event(self, event, viewport):
        type = event.type

        if type == "after_flush":
            self._pointer_pos_prev = self._pointer_pos.copy()
            self._key_down.clear()
            self._key_up.clear()
            self._pointer_down.clear()
            self._pointer_up.clear()
            self._pointer_delta.clear()
            self._wheel_delta.clear()

        elif type.startswith("pointer_"):
            pointer_id = event.pointer_id

            if type == "pointer_move":
                x, y = event.x, event.y
                self._pointer_pos[pointer_id] = (x, y)

                # prevent emitting a bogus delta when we don't have a previous
                # position for the mouse
                if (prev := self._pointer_pos_prev.get(pointer_id)) is not None:
                    prev_x, prev_y = prev
                    self._pointer_delta[pointer_id] = (
                        x - prev_x,
                        y - prev_y,
                    )
            elif type == "pointer_down":
                key = (pointer_id, event.button)
                self._pointer_state[key] = True
                self._pointer_down[key] = True
            elif type == "pointer_up":
                key = (pointer_id, event.button)
                self._pointer_state[key] = False
                self._pointer_up[key] = True

        elif type.startswith("key_"):
            key = event.key.lower()

            if type == "key_down":
                self._key_state[key] = True
                self._key_down[key] = True
            elif type == "key_up":
                self._key_state[key] = False
                self._key_up[key] = True

        elif type == "wheel":
            self._wheel_delta[event.pointer_id] = event.dy

    def key_down(self, key):
        """Returns True if the key_down event was received in the current frame"""
        return self._key_down.get(key.lower(), False)

    def key_up(self, key):
        """Returns True if the key_up event was received in the current frame"""
        return self._key_up.get(key.lower(), False)

    def key(self, key):
        """Returns True if the key is currently down"""
        return self._key_state.get(key.lower(), False)

    def pointer_button_down(self, button, pointer=0):
        """Returns True if the pointer_down event was received in the current frame"""
        return self._pointer_down.get((pointer, button), False)

    def pointer_button_up(self, button, pointer=0):
        """Returns True if the pointer_up event was received in the current frame"""
        return self._pointer_up.get((pointer, button), False)

    def pointer_button(self, button, pointer=0):
        """Returns True if the pointer is currently down"""
        return self._pointer_state.get((pointer, button), False)

    def pointer(self, pointer=0):
        """Returns the current position of a pointer"""
        return self._pointer_pos.get(pointer, (0, 0))

    def pointer_delta(self, pointer=0):
        """Returns the movement of a pointer since the last frame"""
        return self._pointer_delta.get(pointer, (0, 0))

    def wheel(self, pointer=0):
        """Returns the wheel delta of a pointer since the last frame"""
        return self._wheel_delta.get(pointer, 0)
