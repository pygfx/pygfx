from ._cmdata_mpl import (
    _viridis_data,
    _plasma_data,
    _inferno_data,
    _magma_data,
    _cividis_data,
)

from ..resources import Texture


__all__ = ["viridis", "plasma", "inferno", "magma", "cividis"]


viridis = Texture(_viridis_data, dim=1).get_view()
plasma = Texture(_plasma_data, dim=1).get_view()
inferno = Texture(_inferno_data, dim=1).get_view()
magma = Texture(_magma_data, dim=1).get_view()
cividis = Texture(_cividis_data, dim=1).get_view()
