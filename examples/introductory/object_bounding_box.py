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

try:
    # modify this line if your model is located elsewhere
    model_dir = Path(__file__).parents[1] / "models"
except NameError:
    # compatibility with sphinx-gallery
    model_dir = Path(os.getcwd()).parent / "models"


################################################################################
# Once the path is set correctly, you can use the model as follows:

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import trimesh
import pygfx as gfx


teapot = trimesh.load(model_dir / "teapot.stl")

scene = gfx.Scene()
scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

mesh = gfx.Mesh(
    gfx.geometry_from_trimesh(teapot),
    gfx.MeshPhongMaterial(),
)
mesh.rotation.set_from_euler(gfx.linalg.Euler(0.71, 0.91))
scene.add(mesh)

box_world = gfx.BoxHelper(color="red")
box_world.set_transform_by_object(mesh)
scene.add(box_world)

box_local = gfx.BoxHelper(thickness=2, color="green")
box_local.set_transform_by_object(mesh, space="local")
mesh.add(box_local)  # note that the parent is `mesh` here, not `scene`


if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene, up=gfx.linalg.Vector3(0, 0, 1))
