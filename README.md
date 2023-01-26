# pygfx
[![CI](https://github.com/pygfx/pygfx/workflows/CI/badge.svg)](https://github.com/pygfx/pygfx/actions)
[![Documentation Status](https://readthedocs.org/projects/pygfx/badge/?version=latest)](https://pygfx.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/pygfx.svg)](https://badge.fury.io/py/pygfx)

A python render engine targeting Vulkan/Metal/DX12.


<p align="center">
<img src="./docs/_static/readme_sponza.png" alt="drawing" width="200"/>
<img src="./docs/_static/readme_pbr_example.webp" alt="drawing" width="200"/>
<img src="./docs/_static/readme_torus_knot_wire.png" alt="drawing" width="200"/>
</p>
<p align="center">
(Check out the <a href="https://pygfx.readthedocs.io/en/latest/_gallery/index.html">example gallery</a> for more renders.)
</p>

## Installation

```bash
pip install -U pygfx glfw
```

Pygfx needs _some_ window to render to. Glfw is one lightweight option, but
others do exist. If you only use pygfx for offscreen, or notebook rendering you
may choose to omit this. Similarly, you can (of course) swap it for any other
wgpu-compatible canvas, e.g., PyQt, PySide, or wx.

## Example

> **Note**
> A walkthrough of this example can be found in [the guide](https://pygfx.readthedocs.io/en/latest/guide.html#how-to-use-pygfx).

```python
import pygfx as gfx

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)

def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(
            gfx.linalg.Euler(0, 0.01)
        )
    cube.rotation.multiply(rot)

if __name__ == "__main__":
    gfx.show(cube, before_render=animate)

```
<img src="./docs/_static/guide_rotating_cube.gif" alt="drawing" width="400"/>


## Features

> **Note**
> Under construction (TODO)

This is a Python render engine build on top of [WGPU](https://github.com/pygfx/wgpu-py) (instead of OpenGL).

We take a lot of inspiration from ThreeJS, e.g.:

* Materials and Geometry are combined in world objects.
* Decoupled cameras and controllers.
* The code for the render engines is decoupled from the objects, allowing multiple render engines (e.g. wgpu and svg).

Further we aim for a few niceties:
* Proper support for high-res screens.
* Builtin anti-aliasing.
* Custom post-processing steps.
* Support for picking objects and parts within objects.
* (approximate) order-independent transparency (OIT) (not implemented yet).


As an example, see `collections_line.py`: drawing 1000 line objects with 30k points each at 57 FPS (on my laptop).

## License

Pygfx is licensed under the [BSD 2-Clause "Simplified" License](LICENSE). This means:

- :white_check_mark: It is free (and open source) forever :cupid:
- :white_check_mark: You _can_ use it commercially
- :white_check_mark: You _can_ make changes and distribute it
- :x: You _can not_ hold us accountable the results of using pygfx.

## Contributing

> **Note**
> Under construction (TODO)

## Testing examples

> **Note**
> Under construction (TODO)

The test suite is divided into two parts; unit tests for the core, and unit tests for the examples.

* `pytest -v tests` runs the core unit tests.
* `pytest -v examples` tests the examples.
