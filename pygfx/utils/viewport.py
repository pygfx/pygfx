from ..renderers import Renderer


class Viewport:
    """A rectangular area on a renderer.

    Parameters
    ----------
    renderer : Renderer
        The renderer on which the viewport should be defined.
    rect : tuple, [4]
        The viewport rect (x, y, w, h). If None, it is set to the full size of
        the renderer's canvas.

    """

    def __init__(self, renderer, rect=None):
        if not isinstance(renderer, Renderer):
            raise TypeError("Viewport first arg must be a Renderer.")
        self._renderer = renderer
        self.rect = rect

    @classmethod
    def from_viewport_or_renderer(cls, x):
        """Convenience method. Some parts of pygfx that accept a viewport
        might just as well accept a renderer.
        """
        if isinstance(x, Viewport):
            return x
        elif isinstance(x, Renderer):
            return cls(x)
        else:
            raise TypeError("Expected a Viewport or a Renderer.")

    @property
    def renderer(self):
        """A reference to the renderer that this is a viewport of."""
        return self._renderer

    @property
    def logical_size(self):
        """The logical size of the rectangular area described by this viewport."""
        rect = self.rect
        return rect[2], rect[3]

    @property
    def rect(self):
        """The viewport rect (x, y, w, h). Can be set to None to set
        it to the full renderer size.
        """
        return self._get_rect()

    @rect.setter
    def rect(self, rect):
        if rect is None:
            self._rect = None
        else:
            if not (len(rect) == 4):
                raise ValueError("Viewport rect must be 4 numbers, or None.")
            self._rect = tuple(float(v) for v in rect)

    def _get_rect(self):
        """Can be overloaded, e.g. for viewports that define their own rect."""
        if self._rect is None:
            return (0, 0) + self.renderer.logical_size
        else:
            return self._rect

    def is_inside(self, x, y):
        """Get whether the given position is inside the viewport rect."""
        vp = self.rect
        return vp[0] <= x <= vp[0] + vp[2] and vp[1] <= y <= vp[1] + vp[3]

    def render(self, scene, camera, flush=False):
        """A shorthand for ``renderer.render(scene, camera)`` at the appropriate
        viewport. Does not flush by default.
        """
        self.renderer.render(scene, camera, rect=self.rect, flush=flush)
