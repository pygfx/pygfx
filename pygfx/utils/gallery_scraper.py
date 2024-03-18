"""
Utilities for scraping examples to create a gallery.
"""

import os
import re
import time
from pathlib import Path

import imageio.v3 as iio
from sphinx_gallery.scrapers import figure_rst
from wgpu.gui import WgpuCanvasBase

from ..renderers import Renderer
from .show import Display


gallery_pattern = re.compile(r"^# example_gallery:(.+)$", re.MULTILINE)


def find_examples_for_gallery(examples_dir):
    """Find examples to include in the gallery.

    Examples are collected based on the value of the ``example_gallery:``
    comment in the example files:

    * ``example_gallery: hidden`` - not present in the gallery.
    * ``example_gallery: code`` - present only as code (don't run the code when building docs).
    * ``example_gallery: screenshot`` - present in the gallery with a screenshot.
    * ``example_gallery: animate 3s 15fps`` - present in the gallery with an animation (duration and fps are optional).

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
        match = gallery_pattern.search(example_code)
        config = match.group(1).lower().strip() if match else "<missing>"
        if not match:
            examples_to_hide.append(fname)
        elif config == "hidden":
            examples_to_hide.append(fname)
        elif config == "code":
            examples_to_show.append(fname)
        elif config.startswith(("screenshot", "animate")):
            examples_to_run.append(fname)
        else:
            raise RuntimeError(
                f"Unexpected value for '# example_gallery: ' in {fname}: '{config}'."
                + " Expecting 'hidden', 'code', 'screenshot' or 'animate'."
            )

    # Convert to regexp, because that's what Sphinx needs
    patternize = lambda path: (os.sep + path).replace("\\", "\\\\").replace(".", "\\.")
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

    namespace = block_vars["example_globals"]
    path_generator = block_vars["image_path_iterator"]

    # Get canvas to sample the screenshot from
    canvas = select_canvas(namespace)

    # Get gallery config for this block. In config we already select files for
    # inclusion, so if this runs we know that we need a screenshot at least.
    # However, in scripts with multiple blocks, it can still happen that we
    # don't find a match for a specific block. In that case we default to
    # screenshot mode.
    match = gallery_pattern.search(block[1])
    if match:
        config_parts = match.group(1).lower().split()
    else:
        config_parts = ["screenshot"]

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
                raise RuntimeError(f"Unexpected animation option: '{part}'")

        # Store as webp, which is *much* smaller and better looking than gif
        img_filename = Path(next(path_generator)).with_suffix(".webp")
        render_movie(canvas, namespace, img_filename, **config)

    else:
        return

    return figure_rst([img_filename], gallery_conf["src_dir"])


def select_canvas(namespace):
    """Select canvas from the given namespace."""
    # TODO: what can we do here to help find the canvas in e.g. a fastplotlib script?
    canvas = None
    for target_name in ["display", "disp", "renderer", "canvas"]:
        target = namespace.get(target_name, None)
        if target is None:
            pass
        elif isinstance(target, Display):
            canvas = target.canvas
            break
        elif isinstance(target, Renderer):
            canvas = target.target
            break
        elif isinstance(target, WgpuCanvasBase):
            canvas = target
            break
    if canvas is None:
        raise ValueError(
            "Could not find object to sample the screenshot from: need either 'disp', 'renderer' or 'canvas'."
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
        duration=1 / fps,
        loop=loop,
        lossless=lossless,
    )


def split_val_unit(s):
    for i in range(len(s)):
        if s[i] not in "-+0.123456789":
            break
    return float(s[:i]), s[i:]
