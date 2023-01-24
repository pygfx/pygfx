"""
Containers for buffers and textures.

In pygfx, data is stored in buffers and textures. We collectively call these resources.

.. currentmodule:: pygfx.resources

.. autosummary::
    :toctree: resources/
    :template: ../_templates/custom_layout.rst

    Resource
    Buffer
    Texture
    TextureView

"""

# flake8: noqa

from ._buffer import Resource, Buffer
from ._texture import Texture, TextureView

__all__ = ["Resource", "Buffer", "Texture", "TextureView"]
