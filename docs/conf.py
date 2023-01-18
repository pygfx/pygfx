"""Configuration script for Sphinx."""

import os
import sys
from sphinx_gallery.sorting import ExplicitOrder
import wgpu.gui.offscreen


ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

# -- Sphix Gallery Hackz -----------------------------------------------------
# When building the gallery, render offscreen and don't process
# the event loop while parsing the example


def _ignore_offscreen_run():
    wgpu.gui.offscreen.run = lambda: None


os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
_ignore_offscreen_run()

# -- Project information -----------------------------------------------------

project = "pygfx"
copyright = "2021, Almar Klein, Korijn van Golen"
author = "Almar Klein, Korijn van Golen"

# The full version, including alpha/beta/rc tags
# release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autosummary",
    # "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "_gallery",
    "backreferences_dir": "_gallery/backreferences",
    "doc_module": ("pygfx",),
    "image_scrapers": ("pygfx",),
    "subsection_order": ExplicitOrder(
        [
            "../examples/introductory",
            "../examples/feature_demo",
            "../examples/validation",
            "../examples/other",
        ]
    ),
    "remove_config_comments": True,
    # Exclude files in 'other' dir from being executed
    "filename_pattern": r"^((?![\\/]other[\\/]).)*$",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
