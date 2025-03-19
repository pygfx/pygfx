"""Predefined Colormaps.

.. currentmodule:: pygfx.utils.cm

.. autosummary::

    create_colormap
    gray
    viridis
    plasma
    inferno
    magma
    cividis
    cool
    hot
    bone
    copper
    pink
    spring
    summer
    autumn
    winter
    jet

"""

import numpy as np

from ..resources import Texture, TextureMap
from ._cmdata_mpl import (
    _viridis_data,
    _plasma_data,
    _inferno_data,
    _magma_data,
    _cividis_data,
)


_data = {
    # Simple maps
    "gray": [(0, 0, 0, 0), (1, 1, 1, 1)],
    "cool": [(0, 1, 1), (1, 0, 1)],
    "hot": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
    "bone": [(0, 0, 0), (0.333, 0.333, 0.444), (0.666, 0.777, 0.777), (1, 1, 1)],
    "copper": [(0, 0, 0), (1, 0.7, 0.5)],
    "pink": [(0.1, 0, 0), (0.75, 0.5, 0.5), (0.9, 0.9, 0.7), (1, 1, 1)],
    "spring": [(1, 0, 1), (1, 1, 0)],
    "summer": [(0, 0.5, 0.4), (1, 1, 0.4)],
    "autumn": [(1, 0, 0), (1, 1, 0)],
    "winter": [(0, 0, 1), (0, 1, 0.5)],
    "jet": [
        (0, 0, 0.5),
        (0, 0, 1),
        (0, 0.5, 1),
        (0, 1, 1),
        (0.5, 1, 0.5),
        (1, 1, 0),
        (1, 0.5, 0),
        (1, 0, 0),
        (0.5, 0, 0),
    ],
    # Larger maps
    "viridis": _viridis_data,
    "plasma": _plasma_data,
    "inferno": _inferno_data,
    "magma": _magma_data,
    "cividis": _cividis_data,
}

__all__ = sorted([*_data.keys(), "create_colormap"])


def create_colormap(input, n=256):
    """Create a colormap from given data.

    Returns a TextureMap with (at least) n colors.

    If a list or array-like object is given, the values are interpolated to create the map.
    E.g. ``create_colormap([(0, 0, 0), (1, 1, 1)])`` creates the gray colormap.

    If a dict is given, it must have at least one field of "r", "g", "b", "a".
    Each value is a sequence of 2 values (i.e. an Nx2 array-like), where the
    first value specifies the index (0..1), and the second the value for that color.
    """

    if isinstance(input, dict):
        # Init data, alpha 1
        colormap_data = np.zeros((n, 4), "f4")
        colormap_data[:, 3] = 1.0
        # For each channel ...
        for channel_index in range(4):
            channe_name = "rgba"[channel_index]
            try:
                values = input[channe_name]
            except KeyError:
                continue  # this color not in dict
            # Get value list and check
            values = np.asarray(values)
            if not (values.ndim == 2 and values.shape[1] == 2):
                raise ValueError(
                    "The values in the dict to convert to a colormap must be Nx2."
                )
            # Init interpolation
            data = np.zeros((len(values),), dtype="f4")
            x = np.linspace(0.0, 1.0, n)
            xp = np.zeros((len(values),), dtype="f4")
            # Insert values
            for i, el in enumerate(values):
                xp[i] = el[0]
                data[i] = el[1]
            # Interpolate
            colormap_data[:, channel_index] = np.interp(x, xp, data)

    else:
        # Input is list or array-like
        data = np.array(input, dtype="f4")
        if not (data.ndim == 2 and data.shape[1] in [3, 4]):
            raise ValueError("Colormap entries must have 3 or 4 elements.")
        if data.shape[1] == 3:
            data = np.hstack([data, np.ones((data.shape[0], 1), "f4")])

        # Apply interpolation (if required)
        if data.shape[0] == n:
            colormap_data = data
        else:
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, data.shape[0])
            colormap_data = np.zeros((n, 4), "f4")
            for i in range(4):
                colormap_data[:, i] = np.interp(x, xp, data[:, i])

    return TextureMap(Texture(colormap_data, dim=1), filter="nearest", wrap="clamp")


def __dir__():
    return __all__


def __getattr__(name):
    if name not in _data:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    colormap = create_colormap(_data[name])
    globals()[name] = colormap  # prevent creating it again
    return colormap
