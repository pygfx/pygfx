"""
Short utilities to quickly visualize scenes without any boilerplate.
"""
from ..renderers import WgpuRenderer
from ..cameras import PerspectiveCamera
from ..controls import OrbitControls

from .canvases import get_interactive_canvas_cls


def run(backend, canvas_cls, extras, scene, animate=None, interactive=True):
    """
    Render a given scene and optional animate callback in the given backend GUI layer
    """
    if backend == "qt":
        QtWidgets, _ = extras  # noqa: N806
        app = QtWidgets.QApplication([])

    camera = PerspectiveCamera(70, 16 / 9)
    camera.position.z = 500
    controls = None

    if interactive:
        controls = OrbitControls(camera.position.clone())
        canvas_cls = get_interactive_canvas_cls(
            backend,
            app,
            controls,
            camera,
            canvas_cls,
            extras,
        )

    canvas = canvas_cls()
    renderer = WgpuRenderer(canvas)

    def _animate():
        animate()
        if controls:
            controls.update_camera(camera)
        renderer.render(scene, camera)
        canvas.request_draw()

    canvas.request_draw(_animate)

    if backend == "qt":
        app.exec_()
