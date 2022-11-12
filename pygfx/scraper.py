import imageio.v3 as iio
import pygfx as gfx
from sphinx_gallery.scrapers import figure_rst
from wgpu.gui.offscreen import WgpuCanvas


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

    current_scene = None
    current_camera = None
    for var in block_vars["example_globals"].values():
        if isinstance(var, gfx.Scene):
            current_scene = var

        if isinstance(var, gfx.Camera):
            current_camera = var

    if current_scene is None or current_camera is None:
        return ""  # nothing to do

    renderer = gfx.WgpuRenderer(WgpuCanvas())
    renderer.render(current_scene, current_camera)

    path_generator = block_vars["image_path_iterator"]
    img_path = next(path_generator)
    iio.imwrite(img_path, renderer.snapshot())

    return figure_rst([img_path], gallery_conf["src_dir"])


def _get_sg_image_scraper():
    return pygfx_scraper
