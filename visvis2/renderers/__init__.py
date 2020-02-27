# flake8: noqa


class Renderer:
    """ Base class for other renderers. A renderer takes a figure,
    collect data that describe how it should be drawn, and then draws it.
    """

    pass


from .wgpu import WgpuRenderer
from .svg import SvgRenderer
