"""This can probably be written much more cleanly..."""


def get_interactive_canvas_cls(backend, app, controls, camera, canvas_cls, extras):
    if backend == "qt":
        _, QtCore = extras  # noqa: N806

        class WgpuCanvasWithInputEvents(canvas_cls):
            _drag_modes = {QtCore.Qt.RightButton: "pan", QtCore.Qt.LeftButton: "rotate"}
            _mode = None

            def wheelEvent(self, event):  # noqa: N802
                controls.zoom(2 ** (event.angleDelta().y() * 0.0015))

            def mousePressEvent(self, event):  # noqa: N802
                mode = self._drag_modes.get(event.button(), None)
                if self._mode or not mode:
                    return
                self._mode = mode
                drag_start = (
                    controls.pan_start if self._mode == "pan" else controls.rotate_start
                )
                drag_start((event.x(), event.y()), self.get_logical_size(), camera)
                app.setOverrideCursor(QtCore.Qt.ClosedHandCursor)

            def mouseReleaseEvent(self, event):  # noqa: N802
                if self._mode and self._mode == self._drag_modes.get(
                    event.button(), None
                ):
                    self._mode = None
                    drag_stop = (
                        controls.pan_stop
                        if self._mode == "pan"
                        else controls.rotate_stop
                    )
                    drag_stop()
                    app.restoreOverrideCursor()

            def mouseMoveEvent(self, event):  # noqa: N802
                if self._mode is not None:
                    drag_move = (
                        controls.pan_move
                        if self._mode == "pan"
                        else controls.rotate_move
                    )
                    drag_move((event.x(), event.y()))

    return WgpuCanvasWithInputEvents
