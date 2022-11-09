"""Configuration script for Sphinx."""

import os
import sys


ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT_DIR)

import pygfx  # noqa: E402


def _check_objects_are_documented():

    with open(os.path.join(ROOT_DIR, "docs", "reference.rst"), "rb") as f:
        text = f.read().decode()

    # Find what classes are documented
    objects_in_docs = set()
    for line in text.splitlines():
        if line.startswith(".. autoclass::"):
            name = line.split("::")[-1].strip()
            if name.startswith("pygfx."):
                name = name[6:]
            objects_in_docs.add(name)

    # Check world objects
    for name in dir(pygfx):
        ob = getattr(pygfx, name)
        if isinstance(ob, type) and issubclass(ob, pygfx.WorldObject):
            if name not in objects_in_docs:
                raise RuntimeError(f"World object {name} is not documented.")


_check_objects_are_documented()


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
    "image_scrapers": ("pygfx",),
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
