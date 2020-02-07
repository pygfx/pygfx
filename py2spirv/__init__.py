"""
A Python to SpirV compiler.

References:
* https://www.khronos.org/registry/spir-v/
"""

__version__ = "0.2.0"


from ._module import SpirVModule
from ._compiler_raw import bytes2spirv, file2spirv
from ._compiler_pybc import python2spirv
# from ._compiler_wasl import wasl2spirv
from ._compiler_glsl import glsl2spirv
from ._types import vec2, vec3, vec4  # todo: and more
