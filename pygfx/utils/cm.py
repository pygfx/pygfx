"""Predefined Colormaps.

The following colormaps are currently available:

.. currentmodule:: pygfx.utils.cm

.. autosummary::

    viridis
    plasma
    inferno
    magma
    cividis

"""

from ._cmdata_mpl import (
    _viridis_data,
    _plasma_data,
    _inferno_data,
    _magma_data,
    _cividis_data,
)

from ..resources import Texture, TextureMap


__all__ = ["cividis", "inferno", "magma", "plasma", "viridis"]


viridis = TextureMap(Texture(_viridis_data, dim=1), filter="linear", wrap="clamp")
plasma = TextureMap(Texture(_plasma_data, dim=1), filter="linear", wrap="clamp")
inferno = TextureMap(Texture(_inferno_data, dim=1), filter="linear", wrap="clamp")
magma = TextureMap(Texture(_magma_data, dim=1), filter="linear", wrap="clamp")
cividis = TextureMap(Texture(_cividis_data, dim=1), filter="linear", wrap="clamp")
