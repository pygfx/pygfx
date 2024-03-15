"""
The purpose of a renderer is to render (i.e. draw) a scene to a canvas or
texture. It also provides picking, defines the anti-aliasing parameters, and any
post processing effects.

Classes
-------

.. autoclass:: pygfx.renderers.Renderer
    :members:

The system is designed such that different renderer implementations can
co-exist. However, :class:`~wgpu.WgpuRenderer` is likely the
only renderer you'll use.

The ``WgpuRenderer`` is capable of randering all types of objects. Other
renderers will likely be limited to a subset. E.g. the experimental
:class:`~svg.SVGRenderer` for lines, points and text.

Details
-------

A renderer is directly associated with its target and can only render to that
target. Different renderers can render to the same target though.

A renderer provides a ``.render()`` method that can be called one or more times
to render scenes. This creates a visual representation that is stored
internally, and is finally rendered into its render target (the canvas or
texture)::

                              __________
                             | blender  |
    [scenes] -- render() --> |  state   | -- flush() --> [target]
                             |__________|

The internal representation is managed by the blender object. The internal
render textures are typically at a higher resolution to reduce aliasing (SSAA).
The blender has auxiliary buffers such as a depth buffer, pick buffer, and
buffers for transparent fragments. Depending on the blend mode, a single render
call may consist of multiple passes (to deal with semi-transparent fragments).

The flush-step resolves the internal representation into the target texture or
canvas, averaging neighbouring fragments for anti-aliasing.

"""

# flake8: noqa


class Renderer:
    """Base (abstract) renderer class that all renderers inherit from."""

    def render(self, scene, camera):
        """The method to call to render a scene from a camera viewpoint."""
        raise NotImplementedError()


from .wgpu import WgpuRenderer, print_wgpu_report
from .svg import SvgRenderer
