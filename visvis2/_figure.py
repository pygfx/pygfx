import sys
import traceback
import asyncio

import wgpu
from PyQt5 import QtCore, QtGui, QtWidgets
import qasync

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

    # def _on_paint(self):
    #     try:
    #         self._renderer.draw_frame(self)
    #     except Exception:
    #         # Enable PM debuging
    #         sys.last_type, sys.last_value, sys.last_traceback = sys.exc_info()
    #         msg = str(sys.last_value)
    #         msgh = hash(msg)
    #         if msgh in self._err_hashes:
    #             count = self._err_hashes[msgh] + 1
    #             self._err_hashes[msgh] = count
    #             shortmsg = msg.split("\n", 1)[0].strip()[:50]
    #             sys.stderr.write(f"Error in draw again ({count}): {shortmsg}\n")
    #         else:
    #             self._err_hashes[msgh] = 1
    #             sys.stderr.write(f"Error in draw: " + msg.strip() + "\n")
    #             traceback.print_last(6)


# class QtFigureWidget(QtGui.QWindow):
class QtFigureWidget(QtWidgets.QWidget):
    def __init__(self, parent, figure):
        super().__init__(parent)
        self._figure = figure
        self._surface_id = None

        self.setAttribute(QtCore.Qt.WA_PaintOnScreen, True)
        self.setAutoFillBackground(False)

        self.show()

    def _visvis_set_size(self, width, height):
        self.resize(width, height)

    def _visvis_set_title(self, title):
        self.setWindowTitle(title)

    def _visvis_get_surface_id(self, ctx):  # called by renderer
        # Memoize
        if self._surface_id is not None:
            return self._surface_id

        if sys.platform.startswith("win"):
            # Use create_surface_from_windows_hwnd
            hwnd = wgpu.wgpu_ffi.ffi.cast("void *", int(self.winId()))
            hinstance = wgpu.wgpu_ffi.ffi.NULL
            surface_id = ctx.create_surface_from_windows_hwnd(hinstance, hwnd)

        elif sys.platform.startswith("linux"):
            # Use create_surface_from_xlib
            raise NotImplementedError("Linux")

        elif sys.platform.startswith("darwin"):
            # Use create_surface_from_metal_layer
            raise NotImplementedError("OS-X")

        else:
            raise RuntimeError("Unsupported platform")

        self._surface_id = surface_id
        return surface_id

    def _visvis_get_size(self):
        return self.width(), self.height()

    def paintEvent(self, event):
        self._figure._on_paint()

    def paintEngine(self):
        # https://doc.qt.io/qt-5/qt.html#WidgetAttribute-enum  WA_PaintOnScreen
        return None


app = QtWidgets.QApplication([])


# %% Integrate QT with asyncio


if False:
    # The qasync way, probably the best way, but needs a new event loop, so
    # does not integrate so well (yet) with IDE's.
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # An experimental Pyzo thing I hacked together to switch loops
    if hasattr(asyncio, "integrate_with_ide"):
        asyncio.integrate_with_ide(loop, run=False)
else:
    # The quick-n-dirty way, simple and effective, but this limits the
    # rate in which qt can process events. If we could get an event
    # when qt has pending events, this might actually be effective.
    async def _keep_qt_alive():
        while True:
            await asyncio.sleep(0.01)
            app.flush()
            app.processEvents()

    asyncio.get_event_loop().create_task(_keep_qt_alive())
