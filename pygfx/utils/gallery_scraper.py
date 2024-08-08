"""
Utilities for scraping examples to create a gallery.
"""

import os
import re
import time
from pathlib import Path

import imageio.v3 as iio
from sphinx_gallery.scrapers import figure_rst
from sphinx_gallery.py_source_parser import extract_file_config
from wgpu.gui import WgpuCanvasBase

from ..renderers import Renderer


gallery_comment_pattern = re.compile(r"^# *sphinx_gallery_pygfx_docs *\=(.+)$", re.M)


def find_examples_for_gallery(examples_dir):
    """Find examples to include in the gallery.

    Examples are collected based on the value of the ``sphinx_gallery_pygfx_docs``
    comment in the example files:

    * If the comment is not found, the example is excluded.
    * ``sphinx_gallery_pygfx_docs = 'hidden'`` - not present in the gallery.
    * ``sphinx_gallery_pygfx_docs = 'code'`` - present only as code (don't run the code when building docs).
    * ``sphinx_gallery_pygfx_docs = 'screenshot'`` - present in the gallery with a screenshot.
    * ``sphinx_gallery_pygfx_docs = 'animate 3s 15fps'`` - present in the gallery with an animation (duration and fps are optional).

    Returns a dict that can be merged with the sphinx_gallery_conf.

    The Sphinx gallery does not have very good support to configure on a
    per-file basis. This utility function helps collect a lists of files, and
    pour these into regexp's for Sphinx to consume.
    """

    examples_dir = Path(examples_dir).absolute()

    examples_to_hide = []
    examples_to_show = []  # This is Sphinx' default; not actually used.
    examples_to_run = []

    # Collect files
    for filename in examples_dir.glob("**/*.py"):
        fname = str(filename.relative_to(examples_dir))
        example_code = filename.read_text(encoding="UTF-8")
        config = get_example_config(example_code)
        if config is None:
            examples_to_hide.append(fname)  # ignore files not having the comment
        elif not isinstance(config, str):
            raise RuntimeError(
                f"In '{fname}' expected sphinx_gallery_pygfx_docs to be a string."
            )
        elif config == "hidden":
            examples_to_hide.append(fname)
        elif config == "code":
            examples_to_show.append(fname)
        elif config.startswith(("screenshot", "animate")):
            examples_to_run.append(fname)
        else:
            raise RuntimeError(
                f"In '{fname}' got unexpected value for sphinx_gallery_pygfx_docs: '{config}'."
                + " Expecting 'hidden', 'code', 'screenshot' or 'animate'."
            )

    def patternize(path):
        return (os.sep + path).replace("\\", "\\\\").replace(".", "\\.")

    # Convert to regexp, because that's what Sphinx needs
    hide_pattern = "(" + "|".join(patternize(x) for x in examples_to_hide) + ")"
    run_pattern = "(" + "|".join(patternize(x) for x in examples_to_run) + ")"

    # Return dict that can be merged with the sphinx_gallery_conf
    return {
        "examples_dirs": str(examples_dir),
        "ignore_pattern": hide_pattern,
        "filename_pattern": run_pattern,
    }


def pygfx_scraper(block, block_vars, gallery_conf, **kwargs):
    """Scrape pygfx images and animations.

    Parameters
    ----------
    block : tuple
        A tuple containing the (label, content, line_number) of the block.
    block_vars : dict
        Dict of block variables.
    gallery_conf : dict
        Contains the configuration of Sphinx-Gallery
    **kwargs : dict
        Additional keyword arguments to pass to
        :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
        The ``format`` kwarg in particular is used to set the file extension
        of the output file (currently only 'png', 'jpg', and 'svg' are
        supported).

    Returns
    -------
    rst : str
        The ReSTructuredText that will be rendered to HTML containing
        the images. This is often produced by :func:`figure_rst`.
    """

    src_file = block_vars["src_file"]
    namespace = block_vars["example_globals"]
    path_generator = block_vars["image_path_iterator"]

    # Get gallery config for this block. In config we already select files for
    # inclusion, so if this runs we know that we need a screenshot at least.
    # However, in scripts with multiple blocks, it can still happen that we
    # don't find a match for a specific block. In that case we default to
    # screenshot mode.
    config = get_example_config(block[1])
    if not config:
        return ""

    # Get canvas to sample the screenshot from
    canvas = select_canvas(src_file, namespace)

    config_parts = config.split()

    if config_parts[0] == "screenshot":

        # Configure
        config = {
            "lossless": True,
        }

        # Store as webp, even when lossless its ~50% of png
        img_filename = Path(next(path_generator)).with_suffix(".webp")
        render_screenshot(canvas, namespace, img_filename, **config)

    elif config_parts[0] == "animate":

        # Configure
        config = {
            "duration": 3.0,  # total duration of the animation, in seconds
            "fps": 20,  # frames per second
            "loop": 0,  # how many loops to show: 0 means loop forever
            "lossless": False,
        }
        for part in config_parts[1:]:
            val, unit = split_val_unit(part)
            if unit == "s":
                config["duration"] = val
            elif unit == "fps":
                config["fps"] = val
            else:
                raise RuntimeError(
                    f"In '{src_file}' got unexpected animation option: '{part}'"
                )

        # Store as webp, which is *much* smaller and better looking than gif
        img_filename = Path(next(path_generator)).with_suffix(".webp")
        render_movie(canvas, namespace, img_filename, **config)

    else:
        return ""

    return figure_rst([img_filename], gallery_conf["src_dir"])


def get_example_config(example_code):
    file_conf = extract_file_config(example_code)
    return file_conf.get("pygfx_docs", None)


def select_canvas(fname, namespace):
    """Select canvas from the given namespace."""
    canvas = None
    # Try get directly from canvas or renderer
    for target_name in ["renderer", "canvas"]:
        target = namespace.get(target_name, None)
        if target is None:
            pass
        elif isinstance(target, Renderer):
            canvas = target.target
            break
        elif isinstance(target, WgpuCanvasBase):
            canvas = target
            break
    # Try getting from proxy objects
    if canvas is None:
        for target_name in ["display", "disp", "plot", "figure"]:
            target = namespace.get(target_name, None)
            if hasattr(target, "canvas") and isinstance(target.canvas, WgpuCanvasBase):
                canvas = target.canvas
                break
    # Found?
    if canvas is None:
        ns_keys = set(n for n in namespace.keys() if not n.startswith("_"))
        raise ValueError(
            f"In '{fname}' could not find object to sample the screenshot from."
            + f" Need either 'disp', 'renderer' or 'canvas'. Got namespace {ns_keys}"
        )
    return canvas


def render_screenshot(canvas, namespace, filename, *, lossless):
    # Render frame
    frame = canvas.draw()
    # Write image
    iio.imwrite(
        filename,
        frame,
        lossless=lossless,
    )


def render_movie(canvas, namespace, filename, *, duration, fps, loop, lossless):
    # Render frames
    frames = []
    n_frames = int(duration * fps)
    t0 = time.perf_counter()
    for i in range(n_frames):
        namespace["perf_counter"] = lambda: t0 + i / fps
        frames.append(canvas.draw())
    # Write the video
    iio.imwrite(
        filename,
        frames,
        duration=round((1 / fps) * 1000),
        loop=loop,
        lossless=lossless,
    )


def split_val_unit(s):
    for i in range(len(s)):
        if s[i] not in "-+0.123456789":
            break
    return float(s[:i]), s[i:]
