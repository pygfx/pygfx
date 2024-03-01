from ..utils.viewport import Viewport


class Input:
    def __init__(self, register_events=None):
        self.key_down = {}
        self.key_down = {}
        if register_events is not None:
            self.register_events(register_events)

    def register_events(self, viewport_or_renderer):
        viewport = Viewport.from_viewport_or_renderer(viewport_or_renderer)
        viewport.renderer.add_event_handler(
            lambda event: self.handle_event(event, viewport),
            # "pointer_down",
            # "pointer_move",
            # "pointer_up",
            "key_down",
            "key_up",
        )

    def handle_event(self, event, viewport):
        type = event.type

        if type.startswith("key_"):
            # modifiers = {m.lower() for m in event.modifiers}
            # modifiers.discard(event.key.lower())

            if event.type == "key_down":
                self.key_down[event.key.lower()] = True
            elif event.type == "key_up":
                self.key_down[event.key.lower()] = False

    def is_key_down(self, key):
        return self.key_down.get(key.lower(), False)
