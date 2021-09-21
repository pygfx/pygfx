"""
Quickly visualize scenes without any boilerplate.
"""
from .backends import get_backend
from .loop import run


__all__ = ("show",)


def show(scene, animate=None, backend=None, interactive=True):
    backend, canvas_cls, extras = get_backend(name=backend)
    run(backend, canvas_cls, extras, scene, animate=animate, interactive=interactive)
