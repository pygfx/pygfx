""" wgpu renderer namespace.
"""

# flake8: noqa


# Import stuff that people need who create custom shaders, so they can import from pygfx.renderers.wgpu
from ...objects._base import RenderMask
from .engine.utils import (
    registry,
    register_wgpu_render_function,
    gpu_caches,
    to_vertex_format,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
)
from .engine.shared import (
    Shared,
    get_shared,
    stdinfo_uniform_type,
    print_wgpu_report,
    enable_wgpu_features,
)
from .engine.renderer import WgpuRenderer
from .engine.pipeline import Binding

# Shader classes
from .wgsl import load_wgsl
from .shader.base2 import WorldObjectShader
from .shader._shaderlib import shaderlib

# Load shaders submodules, so that all our builtin shaders become a vailable
from . import shaders
