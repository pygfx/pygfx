<h1 align="center"><img src="docs/_static/pygfx_with_name.svg" width="600"/></h1>

[![CI ](https://github.com/pygfx/pygfx/workflows/CI/badge.svg)
](https://github.com/pygfx/pygfx/actions)
[![Documentation Status
](https://readthedocs.org/projects/pygfx/badge/?version=latest)
](https://pygfx.readthedocs.io/en/latest/?badge=latest)
[![PyPI version ](https://badge.fury.io/py/pygfx.svg)
](https://badge.fury.io/py/pygfx)

A python render engine targeting Vulkan/Metal/DX12.

<p align="center">
<img src="./docs/_static/readme_sponza.png" alt="drawing" width="200"/>
<img src="./docs/_static/readme_pbr_example.webp" alt="drawing" width="200"/>
<img src="./docs/_static/readme_torus_knot_wire.png" alt="drawing" width="200"/>
</p>
<p align="center">
[<a href="https://pygfx.readthedocs.io/en/latest/guide.html">User Guide</a>]
[<a href="https://pygfx.readthedocs.io/en/latest/_gallery/index.html">Example Gallery</a>]
[<a href="https://pygfx.readthedocs.io/en/latest/reference.html">API Reference</a>]
</p>

## Installation

```bash
pip install -U pygfx glfw
```

To work correctly, pygfx needs _some_ window to render to. Glfw is one
lightweight option, but there are others, too. If you use a different
wgpu-compatible window manager or only render offscreen you may choose to omit
glfw. Examples of alternatives include: jupyter_rfb (rendering in jupyter),
PyQt, PySide, or wx.

In addition there are some platform
requirements, see the [wgpu docs](https://wgpu-py.readthedocs.io/en/stable/start.html). In
essence, you need modern (enough) graphics drivers, and `pip>=20.3`.

## Usage Example

> **Note**
> A walkthrough of this example can be found in [the
> guide](https://pygfx.readthedocs.io/en/latest/guide.html#how-to-use-pygfx).

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


## Feature Highlights
Some of pygfx's key features are:

- SDF based text rendering ([example](
  https://pygfx.readthedocs.io/en/latest/_gallery/feature_demo/text_contrast.html))
- order-independent transparency (OIT) ([example](
  https://pygfx.readthedocs.io/en/latest/_gallery/feature_demo/transparency2.html))
- lights, shadows, and physically based rendering (PBR) ([example](
  https://pygfx.readthedocs.io/en/latest/_gallery/feature_demo/pbr.html))
- event system with built-in picking ([example](
  https://pygfx.readthedocs.io/en/latest/_gallery/feature_demo/picking_points.html))
- texture and color mapping supporting 1D, 2D and 3D data ([example](
  https://pygfx.readthedocs.io/en/latest/_gallery/feature_demo/colormap_channels.html))


And many more! Check out our [feature demos](
https://pygfx.readthedocs.io/en/latest/_gallery/index.html) in the docs.

## About pygfx

pygfx is a ThreeJS inspired graphics library that uses WGPU (the successor of
OpenGL) to provide GPU acceleration to rendering workloads. It is mature enough
to serve as a general-purpose rendering engine (Yes, you _can_ write a game with
it.) while being geared towards scientific and medical visualization. Thanks to
its low level of abstraction it is flexible and can be adapted to various
use-cases. In other words, pygfx emphasizes on hackability and correctness while
maintaining the level of performance you would expect from a GPU accelerated
library.

## License

Pygfx is licensed under the [BSD 2-Clause "Simplified" License](LICENSE). This means:

- :white_check_mark: It is free (and open source) forever. :cupid:
- :white_check_mark: You _can_ use it commercially.
- :white_check_mark: You _can_ distribute it and freely make changes.
- :x: You _can not_ hold us accountable for the results of using pygfx.

## Contributing
We use a pull request (PR) based workflow similar to many other open-source
libraries in the python ecosystem. You can read more about this workflow
[here](https://docs.github.com/en/get-started/quickstart/github-flow);
if you have previously contributed to open-source, a lot of this will look
familiar already.

### Development Install
To get a working dev install of pygfx you can use the following steps:

```bash
# Click the Fork button on GitHub and navigate to your fork
git clone <address_of_your_fork>
cd pygfx
# if you use a venv, create and activate it
pip install -e .[dev,docs,examples]
pytest
```

### Testing

The test suite is divided into two parts; unit tests for the core, and unit
tests for the examples.

* `pytest -v tests` runs the core unit tests.
* `pytest -v examples` tests the examples.
