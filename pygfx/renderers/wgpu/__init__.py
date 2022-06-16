# flake8: noqa

from ._utils import registry, register_wgpu_render_function
from ._renderer import stdinfo_uniform_type
from ._renderer import WgpuRenderer
from . import meshrender

# from . import linerender
# from . import pointsrender
# from . import imagerender
# from . import volumerender
# from . import backgroundrender

"""
Below is the high level model in which the visualization is described.
There is the obvious world object, material, and geometry. There also
is global data where the renderer stores camera transform, and canvas
sizes (in a single global uniform). And then there is the things defined
by the environment (i.e. scene) like lights and shadows. Renderer
specific state is also part of the environment. We cannot really put
these in a uniform, since they are dynamic and/or affect the render
targets and thus the pipeline.

                           UBuffer
                              ▲
           UBuffer            │
              ▲           ┌───┴────┐      ┌──────────────┐    ┌─────────────┐
              │        ┌──┤Material│      │Environment:  │    │Global:      │
        ┌─────┴─────┐  │  └────────┘      │- blend mode  │    │- glyph atlas│
        │WorldObject├──┤                  │- light slots │    │- stdinfo    │
        └───────────┘  │  ┌────────┐      │- shadow maps │    │             │
                       └──┤Geometry│      │- clip planes?│    └──────┬──────┘
                          └───┬────┘      └──────┬───────┘           │
                              │                  │                   ▼
                              ▼                  ▼                UBuffer
                            Buffers &        Affects wgsl         and Texture
                            Textures         and pipeline


From all this stuff, the shader object corresponding to a material
produces the things below. Note that these are independent on the environment.
This means that

    ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌─────────────┐
    │ WGSL source │  │ Resources:       │  │ pipeline info │  │ render info │
    │ (templated) │  │ - index buffer   │  │ (dict)        │  │ (dict)      │
    │             │  │ - vertex buffers │  │               │  │             │
    │             │  │ - bindings       │  │               │  │             │
    └─────────────┘  └──────────────────┘  └───────────────┘  └─────────────┘


Eventually, we create something like the below, that we can feed to wgpu.

                     ┌────────────────┐  ┌──────────────────────────┐  ┌───────────────────────┐
   RenderContext ───►│WGSL source     │  │BindGroupLayoutDescriptors│  │VertexLayoutDescriptors│
   specific          └───────┬────────┘  └──────────┬───────────────┘  └──────────┬────────────┘
   templating                │                      │                             │
                     ┌───────▼────────┐  ┌──────────▼─────────┐                   │
                     │WgpuShaderModule│  │WgpuBindGroupLayouts│                   │
                     └───────┬────────┘  └──────────┬─────────┘                   │
                             │                      │                             │
                             └──────────────────┐   │  ┌──────────────────────────┘
                                                │   │  │
                                            ┌───▼───▼──▼───┐         ┌─────────────┐
                                            │ WgpuPipeline ◄─────────┤pipeline info│
                                            └───────┬──────┘         └─────────────┘
                                                    │
                          Ubuffers                  │
                          Buffers   ────────────┐   │
                          Textures              │   │
                             │                  │   │
                             │                  │   │
                      ┌──────▼───────┐      ┌───▼───▼─────────┐      ┌───────────┐
                      │WgpuBindGroups├──────► Dispatch (draw) ◄──────┤render info│
                      └──────────────┘      └─────────────────┘      └───────────┘


## Caching

We keep a global list of world-objects that have changed (affecting the state in the 2nd layer).
World-objects subscribe/unsubscribe to this list when they're setting an attribute that they're tracking. Similarly, textures and buffers are also in this list.

When a renderer is about to render, it flushes this list, updating all world-object render info, and uploading all pending data to buffers and textures.

We can give the object-that-stores-the-render-info a revision nr, so that the renderer
can check whether the object has updated or that the render bundle can be re-used. We can
probably fit this in later.

Updating the render-info must lead to fine-grained invalidation of shader and pipeline to
support the easy swapping etc.



(Figures created with https://asciiflow.com/)
"""
