"""
Resource handlers for GPU objects.

.. autosummary::
    :toctree: texture/

    Resource
    Buffer
    Texture
    TextureView

"""

# flake8: noqa

from ._buffer import Resource, Buffer
from ._texture import Texture, TextureView

__all__ = ["Resource", "Buffer", "Texture", "TextureView"]
