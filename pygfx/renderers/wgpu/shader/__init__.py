"""
This subpackage defines the base shader class.
Basically this is where the standard-wgsl-juggling takes place.


## A note about pygfx shaders

In every render engine that uses (GPU) shaders, one particularly nasty problem
is that of shader composition. I.e. how to write shaders and shader snippets,
such that duplicate code is avoided, shaders can be produced for a wide variety
of parameters (i.e. materials and their properties), and still keep the system
understandable.

The way we've solved this in pygfx is by no means a holy grail, but it does feel
(to us) like a local optimum that's pretty flexible while keeping things simple.

Basically, based on the class of the world object and material, the
corresponding shader class is selected. Shader classes compose shaders from
pieces of wgsl, possibly based on properties of the world object / geometry /
material. Some pieces are standard/common code, some are inlined in Python, and
some are larger shaders loaded from wgsl files. Templating using jinja2 is used
to realize "compile time" choices in the code.
"""

from .base1 import BaseShader  # noqa
from .base2 import WorldObjectShader  # noqa
