# pygfx

A render engine, inspired by ThreeJS, but for Python and targeting Vulkan/Metal/DX12 (via wgpu).


## Introduction

This is a Python render engine build on top of [WGPU](https://github.com/pygfx/wgpu-py) (instead of OpenGL).

We take a lot of inspiration from ThreeJS, e.g.:

* Materials and Geometry are combined in world objects.
* No event system, but controls that make it relatively easy to integrate with one.
* Decoupled cameras and controls.
* The code for the render engines is decoupled from the objects, allowing multiple render engines (e.g. wgpu and svg).


## How to build a visialization

See also the examples, they all do something like this:

* Instantiate a renderer and a canvas to render to.
* Create a scene and populate it with world objects.
* Create a camera (and maybe a control).
* Define an  animate function that calls: `renderer.render(scene, camera)`


## On world objects, materials, and geometry

There are a few different world object classes. The class defines
(semantically) the kind of object being drawn, e.g. `Line`, `Image`,
`Mesh`, `Volume`. World objects have a position and orientation in the
scene, and can have children (other world objects), creating a tree.
World objects can also have a geometry and/or a material.

The geometry of an object defines its base data, usually per-vertex
attributes such as positions, normals, and texture coordinates. There
are several pre-defined geometries, most of which simply define certain
3D shapes.

The material of an object defines how an object is rendered. Usually
each WorldObject class has one or more materials associated with it.
E.g. a line can be drawn solid, segmented or with arrows. A volume can
be rendered as a slice, MIP, or something else.


## Installation

```bash
pip install -U https://github.com/pygfx/pygfx/archive/main.zip
```


## Current status

Under development, many things can change. We don't even do releases yet ...
