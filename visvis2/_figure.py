import sys

import wgpu

from . import _renderer


class Figure:
    """ Represents the root rectangular region to draw to.
    """

    def __init__(self, canvas, parent=None, renderer=None):
        self._views = []
        self._widget = canvas  # todo: rename to canvas
        self._err_hashes = {}

        # Check renderer
        if renderer is None:
            self._renderer = _renderer.SurfaceWgpuRenderer()
        else:
            self._renderer = render

        canvas.drawFrame = self._draw_frame

    def _draw_frame(self):
        # Called by canvas
        self._renderer.draw_frame(self)

    @property
    def views(self):
        return self._views

    @property
    def renderer(self):
        return self._renderer

    @property
    def widget(self):
        return self._widget

    @property
    def size(self):
        return self._widget._visvis_get_size()

    def get_surface_id(self, ctx):
        return self._widget._visvis_get_surface_id(ctx)


# # Bit of code that uses qasync to integrate Qt with asyncio
# # The qasync way, probably the best way, but needs a new event loop, so
# # does not integrate so well (yet) with IDE's.
# loop = qasync.QEventLoop(app)
# asyncio.set_event_loop(loop)
#
# # An experimental Pyzo thing I hacked together to switch loops
# if hasattr(asyncio, "integrate_with_ide"):
#     asyncio.integrate_with_ide(loop, run=False)
