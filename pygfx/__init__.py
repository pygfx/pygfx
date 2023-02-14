# flake8: noqa

from . import utils

from .resources import *
from .objects import *
from .geometries import *
from .materials import *
from .cameras import *
from .helpers import *
from .controllers import *

from .renderers import *

from .utils.color import Color
from .utils.load import load_scene
from .utils.show import show, Display
from .utils.viewport import Viewport
from .utils.text import font_manager
from .utils import cm, logger


__version__ = "0.1.11"
version_info = tuple(map(int, __version__.split(".")))

__wgpu_version_range__ = "0.9.0", "0.10.0"


def _test_wgpu_version():
    import wgpu  # noqa

    min_ver, max_ver = __wgpu_version_range__
    min_ver_info = tuple(map(int, min_ver.split(".")))
    max_ver_info = tuple(map(int, max_ver.split(".")))
    detected = f"Detected {wgpu.__version__}, need >={min_ver}, <{max_ver}."
    if wgpu.version_info < min_ver_info:
        logger.error(
            f"Incompatible version of wgpu-py:\n    {detected}\n    To update, use e.g. `pip install -U wgpu`."
        )
    elif wgpu.version_info >= max_ver_info:
        logger.warning(f"Possible incompatible version of wgpu-py:\n    {detected}")


_test_wgpu_version()


def _get_sg_image_scraper():
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
