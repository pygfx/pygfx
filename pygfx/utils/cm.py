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

from ..resources import Texture


__all__ = ["viridis", "plasma", "inferno", "magma", "cividis"]


viridis = Texture(_viridis_data, dim=1)
plasma = Texture(_plasma_data, dim=1)
inferno = Texture(_inferno_data, dim=1)
magma = Texture(_magma_data, dim=1)
cividis = Texture(_cividis_data, dim=1)
