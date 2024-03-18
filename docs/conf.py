"""Configuration script for Sphinx."""

import os
import re
import sys
import shutil
import warnings
from itertools import chain
from pathlib import Path


from sphinx_gallery.sorting import ExplicitOrder
import wgpu.gui.offscreen


ROOT_DIR = Path(__file__).parents[1]  # repo root
EXAMPLES_DIR = ROOT_DIR / "examples"

sys.path.insert(0, str(ROOT_DIR))


# -- Project information -----------------------------------------------------

project = "pygfx"
copyright = "2021-2024, Almar Klein, Korijn van Golen"
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

# Just let autosummary produce a new version each time
shutil.rmtree(os.path.join(os.path.dirname(__file__), "_autosummary"), True)

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Sphix Gallery -----------------------------------------------------

# Force offscreen rendering
os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

# The Sphinx gallery does not have very good support to configure on a per-file
# basis. So we do that ourselves. We collect lists of files, and then pour these
# into regexp's for Sphinx to consume.
examples_to_hide = []
examples_to_show = []  # This is Sphinx' default, added for completenes, but not used.
examples_to_run = []

gallery_pattern = re.compile(r"^# example_gallery:(.+)$", re.MULTILINE)

# Collect files
for filename in EXAMPLES_DIR.glob("**/*.py"):
    fname = str(filename.relative_to(EXAMPLES_DIR))
    if "tests" in filename.parts:
        continue
    example_code = filename.read_text(encoding="UTF-8")
    match = gallery_pattern.search(example_code)
    config = match.group(1).lower().strip() if match else "<missing>"
    if config == "hidden":
        examples_to_hide.append(fname)
    elif config == "code":
        examples_to_show.append(fname)
    elif config.startswith(("screenshot", "animate")):
        examples_to_run.append(fname)
    else:
        examples_to_hide.append(fname)
        warnings.warn(
            f"Unexpected value for '# example_gallery: ' in {fname}: {config}."
        )

# Convert to regexp, because that's what Sphinx needs
patternize = lambda path: (os.sep + path).replace("\\", "\\\\").replace(".", "\\.")
sphinx_hide_pattern = "(" + "|".join(patternize(x) for x in examples_to_hide) + ")"
sphinx_run_pattern = "(" + "|".join(patternize(x) for x in examples_to_run) + ")"

# The gallery conf. See https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "_gallery",
    "backreferences_dir": "_gallery/backreferences",
    "doc_module": ("pygfx",),
    # Tell sphinx to use the scraper in the pygfx lib. This detects pygfx.utils.gallery_scraper.pygfx_scraper()
    "image_scrapers": ("pygfx",),
    # Don't show mini-galleries for these objects, because they include nearly all examples.
    "exclude_implicit_doc": {
        "WgpuRenderer",
        "Resource",
        "WorldObject",
        "Geometry",
        "Material",
        "Controller",
        "Camera",
        "show",
        "Display",
        "Group",
        "Scene",
        "Light",
    },
    # Define order of appearance of the examples
    "subsection_order": ExplicitOrder(
        [
            "../examples/introductory",
            "../examples/feature_demo",
            "../examples/validation",
            "../examples/other",
        ]
    ),
    # Patterns to exclude and run examples, respectively
    "ignore_pattern": sphinx_hide_pattern,
    "filename_pattern": sphinx_run_pattern,
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
html_favicon = "_static/pygfx.ico"
html_logo = "_static/pygfx.svg"
