"""
The engine of the wgpu renderer.

This is the only place in pygfx where we make wgpu function calls.
(Although we may use enums or flags elsewhere.)

----

Below is the high level model in which the visualization is described.
Let's call this level 1. There is the obvious world object, material,
and geometry. There also is global data where the renderer stores camera
transform, and canvas sizes (in a single global uniform). And then there
is the things defined by the environment (i.e. scene) like lights and
shadows. Renderer specific state is also part of the environment.

                           Uniform-Buffer
                              ▲
       Uniform-Buffer         │
              ▲           ┌───┴────┐      ┌──────────────┐    ┌─────────────┐
              │        ┌──┤Material│      │Environment:  │    │Global:      │
        ┌─────┴─────┐  │  └────────┘      │- blend mode  │    │- glyph atlas│
        │WorldObject├──┤                  │- light slots │    │- stdinfo    │
        └───────────┘  │  ┌────────┐      │- shadow maps │    │             │
                       └──┤Geometry│      │- clip planes?│    └──────┬──────┘
                          └───┬────┘      └──────┬───────┘           │
                              │                  │                   ▼
                              ▼                  ▼              Uniform-Buffer
                            Buffers &        Affects wgsl         and Texture
                            Textures         and pipeline

From all this stuff, we create an intermediate representation. Let's
call this level 2. This information is created by the BaseShader
corresponding to a certain material. Note that this object is agnostic
about the environment.

    ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  ┌─────────────┐
    │ WGSL source │  │ Resources:       │  │ pipeline info │  │ render info │
    │ (templated) │  │ - index buffer   │  │ (dict)        │  │ (dict)      │
    │             │  │ - vertex buffers │  │               │  │             │
    │             │  │ - bindings       │  │               │  │             │
    └─────────────┘  └──────────────────┘  └───────────────┘  └─────────────┘

The objects above are not wgpu-specific, but make it easy to create
wgpu-specific objects that we see below (level 3). This step is
performed by the PipelineContainer. Some of these objects match 1-on-1
to the world object, but others (shader module and pipeline) are
specific to the environment. And then there are multiple blend-passes
to create as well. The environment also plays a role in "finalizing"
the final environment-specific objects.

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
                      Uniform-Buffers               │
                          Buffers   ────────────┐   │
                          Textures              │   │
                             │                  │   │
                             │                  │   │
                      ┌──────▼───────┐      ┌───▼───▼─────────┐      ┌───────────┐
                      │WgpuBindGroups├──────► Dispatch (draw) ◄──────┤render info│
                      └──────────────┘      └─────────────────┘      └───────────┘

## Reacting to changes

To keep a PipelineContainer up-to-date we need to do two things. First
any changes in the world object and/or material must be applied. This
is done via the Trackable classes, and affects the second level. Next,
some of these changes invalidate the objects in level 3, so we need to
detect that as well.

Before each draw, all buffers and textures that have pending changes are
flushed.

## Caching

Objects are stored for each specific environment type, by using the
environment's hash. The environment includes a system to detect that
it is no longer used to that all objects related to that environment
can be cleaned up.

We also re-use wgpu objects like shadermodules. If there are a lot of
objects in a scene, its likely that many of these have the same material.

(Figures created with https://asciiflow.com/)
"""
