import importlib


BACKENDS_QT = ("PySide6", "PyQt6", "PySide2", "PyQt5", "PySide", "PyQt4")
BACKENDS = ("qt", "wx", "glfw", "jupyter")


def get_backend(name=None):
    """Automatically detect which GUI layer and canvas can be used"""
    backends = (name,) if name else BACKENDS
    extras = None
    for backend in backends:
        try:
            if backend == "qt":
                for backend_qt in BACKENDS_QT:
                    try:
                        importlib.import_module(f"{backend_qt}.QtCore")
                    except ImportError:
                        pass
            mod = importlib.import_module(f"wgpu.gui.{backend}")
            canvas_cls = getattr(mod, "WgpuCanvas")
            if backend == "qt":
                extras = getattr(mod, "QtWidgets"), getattr(mod, "QtCore")
            return backend, canvas_cls, extras
        except (ImportError, AttributeError):
            pass
    raise ImportError("No compatible backend found")
