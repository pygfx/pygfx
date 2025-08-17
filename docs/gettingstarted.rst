Getting started with PyGfx
==========================

Installation
------------

To install use your favourite package manager, e.g.:

.. code-block::

    pip install -U pygfx

For Pygfx to work, the appropriate GPU drivers should be installed.

* Windows: On Windows 10 and up, you should be ok. If your machine has a dedicated GPU, consider updating your (Nvidia or AMD) drivers.
* MacOS: Need at least 10.13 (High Sierra) to use Metal (Apple's GPU driver).
* Linux: On a modern Linux desktop you should be fine. Maybe ``apt install mesa-vulkan-drivers``.
  For details see https://wgpu-py.readthedocs.io/en/stable/start.html#linux.



Using Pygfx in Jupyter
----------------------

You can use Pygfx in the Jupyter notebook and Jupyter lab. To do so,
use the Jupyter canvas provided by WGPU, and use that canvas as the cell output.

.. code-block:: python

    from rendercanvas.jupyter import RenderCanvas

    canvas = RenderCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)

    ...

    canvas  # cell output

Also see the Pygfx examples `here <https://jupyter-rfb.readthedocs.io/en/stable/examples/>`_.

