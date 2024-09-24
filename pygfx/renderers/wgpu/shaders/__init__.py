"""
All pygfx builtin shaders.

The idea is that these only import from the (public) namespace pygfx.renderers.wgpu.
"""

# ruff: noqa: F401

from . import backgroundshader
from . import gridshader
from . import meshshader
from . import pointsshader
from . import lineshader
from . import imageshader
from . import volumeshader
from . import textshader
