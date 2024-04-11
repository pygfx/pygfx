"""
Show STL File via gfx.show
==========================

Demonstrates show utility with an STL file
"""

################################################################################
# .. note::
#
#   To run this example, you need a model from the source repo's example
#   folder. If you are running this example from a local copy of the code (dev
#   install) no further actions are needed. Otherwise, you may have to replace
#   the path below to point to the location of the model.

import os
from pathlib import Path

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"


################################################################################
# Once the path is set correctly, you can use the model as follows:

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import pygfx as gfx


meshes = gfx.load_mesh(model_dir / "teapot.stl")

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(meshes[0], up=(0, 0, 1))
