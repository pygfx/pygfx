import re
import time
from pathlib import Path

import imageio.v3 as iio
from sphinx_gallery.scrapers import figure_rst
from wgpu.gui import WgpuCanvasBase

from ..renderers import Renderer
from .show import Display


gallery_pattern = re.compile(r"^# example_gallery:(.+)$", re.MULTILINE)


def split_val_unit(s):
    for i in range(len(s)):
        if s[i] not in "-+0.123456789":
            break
    return float(s[:i]), s[i:]


def pygfx_scraper(block, block_vars, gallery_conf, **kwargs):
    """Scrape pygfx images and animations

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

    # Get canvas to sample the screenshot from
    canvas = None
    for target_name in ["display", "disp", "renderer", "canvas"]:
        target = block_vars["example_globals"].get(target_name, None)
        if isinstance(target, Display):
            canvas = target.canvas
            break
        elif isinstance(target, Renderer):
            canvas = target.target
            break
        elif isinstance(target, WgpuCanvasBase):
            canvas = target
            break
        elif target is not None:
            raise ValueError(f"WTF {target}")
    if canvas is None:
        raise ValueError(
            "Need a target to sample the screenshot from: either 'display', 'renderer' or 'canvas'."
        )

    image_paths = []
    config_parts = gallery_pattern.search(block[1]).group(1).lower().split()

    if config_parts[0] == "screenshot":

        # Configure
        lossless = True

        # Render frame
        frame = canvas.draw()

        # Write image
        path_generator = block_vars["image_path_iterator"]
        img_path = Path(next(path_generator)).with_suffix(".webp")
        iio.imwrite(
            img_path,
            frame,
            lossless=True,
        )
        image_paths.append(img_path)

    elif config_parts[0] == "animate":

        # Configure
        duration = 3.0  # total duration of the animation, in seconds
        fps = 20  # frames per second
        loop = 0  # how many loops to show: 0 means loop forever
        lossless = False
        for part in config_parts[1:]:
            val, unit = split_val_unit(part)
            if unit == "s":
                duration = val
            elif unit == "fps":
                fps = val
            else:
                raise RuntimeError(f"Unexpected animation option: '{part}'")

        # Render frames
        frames = []
        n_frames = int(duration * fps)
        t0 = time.perf_counter()
        for i in range(n_frames):
            block_vars["example_globals"]["perf_counter"] = lambda: t0 + i / fps
            frames.append(canvas.draw())

        # Write the video
        path_generator = block_vars["image_path_iterator"]
        img_path = Path(next(path_generator)).with_suffix(".webp")
        iio.imwrite(
            img_path,
            frames,
            duration=1 / fps,
            loop=loop,
            lossless=lossless,
        )
        image_paths.append(img_path)

    return figure_rst(image_paths, gallery_conf["src_dir"])
