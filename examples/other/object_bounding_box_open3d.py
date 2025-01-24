"""
Boundary Boxes
==============

Demonstrates visualizing object bounding boxes
"""

################################################################################
# .. warning::
#     An external model is needed to run this example.
#
# To run this example, you need a model from the source repo's example
# folder. If you are running this example from a local copy of the code (dev
# install) no further actions are needed. Otherwise, you may have to replace
# the path below to point to the location of the model.

import os
from pathlib import Path
from importlib.util import find_spec

if not find_spec("open3d"):
    raise ImportError(
        "The `open3d` library is required for this example: pip install open3d"
    )

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "data"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "data"

################################################################################
# Once the path is set correctly, you can use the model as follows:

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

from open3d import io
import open3d as o3d
import pygfx as gfx
import pylinalg as la

# load teapot with open3d
teapot: o3d.geometry.TriangleMesh = io.read_triangle_mesh(
    str(model_dir / "teapot.stl"), enable_post_processing=True
)

# open3d does not seem to read stl files with normals -> re-compute them
teapot.compute_vertex_normals()

scene = gfx.Scene()
scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

mesh = gfx.Mesh(
    gfx.geometries.geometry_from_open3d(teapot),
    gfx.MeshPhongMaterial(),
)
mesh.local.rotation = la.quat_from_euler((0.71, 0.91), order="XY")
scene.add(mesh)

box_world = gfx.BoxHelper(color="red")
box_world.set_transform_by_object(mesh)
scene.add(box_world)

box_local = gfx.BoxHelper(thickness=2, color="green")
box_local.set_transform_by_object(mesh, space="local")
mesh.add(box_local)  # note that the parent is `mesh` here, not `scene`

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene, up=(0, 0, 1))
