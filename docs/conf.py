"""Configuration script for Sphinx."""

import os
import subprocess
import sys
import shutil
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder
import pygfx
from pygfx.utils.gallery_scraper import find_examples_for_gallery


ROOT_DIR = Path(__file__).parents[1]  # repo root
EXAMPLES_DIR = ROOT_DIR / "examples"

sys.path.insert(0, str(ROOT_DIR))


# -- Project information -----------------------------------------------------

project = "pygfx"
copyright = "2021-2025, Almar Klein, Korijn van Golen"
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
    "sphinx_gallery.gen_gallery",  # comment this to temporarily turn off the gallery for faster builds
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",
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

# -- Read The Docs integration ---------------------------------------------------

# Define the canonical URL for custom domain on RTD
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on RTD
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True


# -- Build wheel so Pyodide examples can use exactly this version of pygfx -----------------------------------------------------

short_version = ".".join(str(i) for i in pygfx.version_info[:3])
wheel_name = f"pygfx-{short_version}-py3-none-any.whl"

# Build the wheel
subprocess.run([sys.executable, "-m", "build", "-nw"], cwd=ROOT_DIR)
wheel_filename = os.path.join(ROOT_DIR, "dist", wheel_name)
assert os.path.isfile(wheel_filename), f"{wheel_name} does not exist"

# Copy into static
# TODO: you can use --outdir on the build command directly. also use the html_static_path in this namespace
print("Copy wheel to _static dir")
shutil.copy(
    wheel_filename,
    os.path.join(ROOT_DIR, "docs", "_static", wheel_name),
)



# -- Sphix Gallery -----------------------------------------------------

## pyodide demos, adapted from wgpu, adapted from rendercanvas... might make sense to put this in the scaper?
iframe_placeholder_rst = """
.. only:: html

    Interactive example
    -------------------

    Try this example in your browser using Pyodide. Might not work with all examples and all devices. Check the output and your browser's console for details.

    .. raw:: html

        <iframe src="./../pyodide.html#example.py"></iframe>
"""
python_files = {}

def add_pyodide_to_examples(app):
    if app.builder.name != "html":
        return

    gallery_dir = ROOT_DIR / "docs" / "_gallery"
    example_files = gallery_dir.glob("**/*.py")

    for py_file in example_files:
        fname = py_file.name
        with open(py_file, "rb") as f:
            py = f.read().decode()
        if fname:
            print("Adding Pyodide example to", fname)
            fname_rst = py_file.with_suffix(".rst")
            # Update rst file
            rst = iframe_placeholder_rst.replace("example.py", fname)
            # we likely don't want append here?
            with open(fname_rst, "ab") as f:
                # skip if it already ends with the placeholder? otherwise the append will keep on appending (we have to hook this into the gen_rst to skip if possible?)
                f.write(rst.encode())
            python_files[py_file.relative_to(gallery_dir)] = py

def add_files_to_run_pyodide_examples(app, exception):
    if app.builder.name != "html":
        return

    gallery_build_dir = os.path.join(app.outdir, "_gallery")

    # Write html file that can load pyodide examples
    with open(
        os.path.join(ROOT_DIR, "docs", "_static", "_pyodide_iframe.html"), "rb"
    ) as f:
        html = f.read().decode()
    html = html.replace('"pygfx"', f'"../_static/{wheel_name}"')
    with open(os.path.join(gallery_build_dir, "pyodide.html"), "wb") as f:
        f.write(html.encode())

    # Write the python files
    for fname, py in python_files.items():
        print("Writing", fname)
        with open(os.path.join(gallery_build_dir, fname), "wb") as f:
            f.write(py.encode())

# Force offscreen rendering
os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

# Suppress "cannot cache unpickable configuration value" for sphinx_gallery_conf
# See https://github.com/sphinx-doc/sphinx/issues/12300
suppress_warnings = ["config.cache"]

# The gallery conf. See https://sphinx-gallery.github.io/stable/configuration.html
sphinx_gallery_conf = {
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
    # Remove any comment that starts with "sphinx_gallery_"
    "remove_config_comments": True,
    # Define order of appearance of the examples
    "subsection_order": ExplicitOrder(
        [
            "../examples/introductory",
            "../examples/feature_demo",
            "../examples/validation",
            "../examples/other",
        ]
    ),
}

sphinx_gallery_conf.update(find_examples_for_gallery(ROOT_DIR / "examples"))


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/pygfx.svg"
html_logo = "_static/pygfx.svg"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "style.css",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pygfx": ("https://docs.pygfx.org/stable", None),
    "wgpu": ("https://wgpu-py.readthedocs.io/en/latest", None),
}


def setup(app):
    app.connect("builder-inited", add_pyodide_to_examples)
    app.connect("build-finished", add_files_to_run_pyodide_examples)
