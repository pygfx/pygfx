class Renderer:
    """ Base class for other renderers. A renderer takes a figure,
    collect data that describe how it should be drawn, and then draws it.
    """

    pass


class SvgRenderer(Renderer):
    """ Render to SVG. Because why not.
    """

    pass


class GlRenderer(Renderer):
    """ Render with OpenGL. This is mostly there to illustrate that it *could* be done.
    WGPU can (in time) also be used to render using OpenGL.
    """

    pass

