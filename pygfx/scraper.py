import imageio.v3 as iio
import pygfx as gfx
from sphinx_gallery.scrapers import figure_rst
from wgpu.gui.offscreen import WgpuCanvas, run


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

    current_scenes = []
    for var in block_vars["example_globals"].values():
        if isinstance(var, gfx.Scene):
            current_scenes.append(var)

    if len(current_scenes) == 0:
        return ''  # nothing to do

    

    print("")

    return figure_rst([], gallery_conf['src_dir'])


def _get_sg_image_scraper():
    return pygfx_scraper