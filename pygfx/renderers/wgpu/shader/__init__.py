"""
This subpackage defines the base shader class. Basically this is where the wgsl
code is generated, composed, mixed and juggled.


## A note about pygfx shaders

In every render engine that uses (GPU) shaders, one particularly nasty problem
is that of shader composition. I.e. how to write shaders and shader snippets,
such that duplicate code is avoided, shaders can be produced for a wide variety
of parameters (i.e. materials and their properties), and still keep the system
understandable.

The way we've solved this in pygfx is by no means a holy grail, but it does feel
(to us) like a local optimum that's pretty flexible while keeping things simple.

Basically, based on the class of the world object and material, the
corresponding shader class is selected. Shader classes produce templated wgsl
(jinja2), which is then transformed into the final shader code using templating
variables, based on properties of the material and geometry, as well as external
features such as the render pass and environment (e.g. lights).

Wgsl templates can be provided in-line or loaded from wgsl files. Jinja2's
include mechanics is used to write snippets that can be used in multiple
places.
"""

from .base import ShaderInterface, BaseShader  # noqa
from .templating import register_wgsl_loader  # noqa
