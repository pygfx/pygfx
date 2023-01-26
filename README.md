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
may choose to omit glfw. Similarly, you can (of course) swap it for any other
wgpu-compatible canvas, e.g., PyQt, PySide, or wx.

## Example

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


## Features
Some of pygfx's key features are:

- Native support for High Resolution screens.
- Builtin anti-aliasing.
- Support for picking objects and parts within objects.

@almarklein: A few suggestions and feature discussions
- fast draws (57 FPS @ 30M vertices from 1k lines on a . . .) (@almarklein: what specs does your laptop have? Also, do we have a polygon count benchmark?)
- (approximate) order-independent transparency (OIT). (is this implemented now?)
- Custom post-processing. (do the examples for this need updating?)
- Easy shader templating
- built atop [WGPU](https://github.com/pygfx/wgpu-py) (>= OpenGL)
- semantically similar to TreeJS (WorldObjects, decoupled rendering backend, ...)

And more ... check out the [feature demos](https://pygfx.readthedocs.io/en/latest/_gallery/index.html) in the docs.

## License

Pygfx is licensed under the [BSD 2-Clause "Simplified" License](LICENSE). This means:

- :white_check_mark: It is free (and open source) forever :cupid:
- :white_check_mark: You _can_ use it commercially
- :white_check_mark: You _can_ make changes and distribute it
- :x: You _can not_ hold us accountable the results of using pygfx.

## Contributing
We use a very similar setup to many other open-source libraries in the python
ecosystem. If you have contributed to open-source in the past, a lot of this
should look familiar.

**GitHub Workflow**
We use a PR-based workflow. Each contributor has his/her own fork of the project
inside of which he/she maintains a branch containing the changes that should be
incorporated into the project. These changes are presented as a [pull request
(PR)](https://github.com/pygfx/pygfx/pulls) on GitHub where we discuss changes,
review code, and eventually merge the work. 

When you begin working on pygfx you would typically do something like:

```bash
# Click the Fork button on GitHub and navigate to your fork
git clone <address_of_your_fork>
cd pygfx
git remote add upstream git@github.com:pygfx/pygfx.git
pip install -e .[dev,docs,examples]
git checkout -b <branch_name>
git push --set-upstream origin <branch_name>
# Go to https://github.com/pygfx/pygfx/pulls and open a new PR
# Make changes and discuss on GH.
```

While your work is ongoing, other PRs may finish and get merged, which will
require you to either merge with main or rebase. There are different ways to do
this, and the most convenient that I know of is

```bash
git fetch upstream main:main
git push main
git merge main
```

## Testing examples

> **Note**
> Under construction (TODO)

The test suite is divided into two parts; unit tests for the core, and unit tests for the examples.

* `pytest -v tests` runs the core unit tests.
* `pytest -v examples` tests the examples.
