"""The Pygfx render engine."""

# ruff: noqa: F401, F403

from ._version import __version__, version_info, repo_dir as _repo_dir
from . import utils

from .resources import *
from .objects import *
from .geometries import *
from .materials import *
from .cameras import *
from .helpers import *
from .controllers import *
from .animation import *

from .renderers import *

from .utils.color import Color
from .utils.load import load_mesh, load_meshes, load_scene
from .utils.load_gltf import (
    load_gltf,
    load_gltf_async,
    load_gltf_mesh,
    load_gltf_mesh_async,
    print_scene_graph,
)
from .utils.show import show, Display
from .utils.viewport import Viewport
from .utils.text import font_manager
from .utils import cm, enums, logger
from .utils.enums import *

# Temp fix for pyinstaller to pick up pylinalg
import pylinalg

del pylinalg


def _get_dependency_version_ranges():
    # The only case where this dependency checking makes sense is for devs
    # using pygfx from a Git repo.
    if not _repo_dir:
        return {}
    # Try import, dont care when this fails (e.g. frozen or py < 3.11)
    try:
        import os, tomllib  # noqa
    except ImportError:
        return {}
    # Load dependency versions
    with open(os.path.join(_repo_dir, "pyproject.toml"), "rb") as fp:
        dependencies = tomllib.load(fp)["project"]["dependencies"]
    # Parse
    limits_per_dependency = {}
    for name_verlimits in dependencies:
        name, _, verlimits = name_verlimits.partition(" ")
        if name in ("wgpu", "pylinalg"):
            min_ver = max_ver = 0
            for lim in verlimits.split(","):
                if lim.startswith(">="):
                    min_ver = tuple(map(int, lim[2:].split(".")))
                elif lim.startswith("<"):
                    max_ver = tuple(map(int, lim[1:].split(".")))
            if min_ver and max_ver:
                limits_per_dependency[name] = min_ver, max_ver
    return limits_per_dependency


def _check_lib_version(libname, pipname):
    import importlib

    if libname not in _dependency_version_ranges:
        return
    lib = importlib.import_module(libname)
    min_ver, max_ver = _dependency_version_ranges[libname]
    detected = f"Detected {lib.__version__}, need >={min_ver}, <{max_ver}."
    if lib.version_info < min_ver:
        logger.error(
            f"Incompatible version of {libname}:\n    {detected}\n    To update, use e.g. `pip install -U {pipname}`."
        )
    elif lib.version_info >= max_ver:
        logger.warning(f"Possible incompatible version of {libname}:\n    {detected}")


_dependency_version_ranges = _get_dependency_version_ranges()
_check_lib_version("wgpu", "wgpu")
_check_lib_version("pylinalg", "pylinalg")


def _get_sg_image_scraper():
    """Hook for sphinx so we can tell it how to generate the gallery."""
    import sphinx_gallery.scrapers
    from .utils.gallery_scraper import pygfx_scraper

    # add webp as supported extension
    sphinx_gallery.scrapers._KNOWN_IMG_EXTS += ("webp",)

    return pygfx_scraper


# Elements of this library are derived from three.js, original license
# at time of writing copied here:
# ---
# The MIT License
#
# Copyright © 2010-2022 three.js authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ---
# End of original license copy.
