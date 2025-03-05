"""
Containers for buffers and textures.

In Pygfx, data is stored in buffers and textures. We collectively call these resources.

.. currentmodule:: pygfx.resources

.. autosummary::
    :toctree: resources/
    :template: ../_templates/custom_layout.rst

    Resource
    Buffer
    Texture
    TextureMap

"""

# ruff: noqa: F401

from ._base import Resource
from ._buffer import Buffer
from ._texture import Texture
from ._texturemap import TextureMap

__all__ = ["Buffer", "Resource", "Texture", "TextureMap"]
