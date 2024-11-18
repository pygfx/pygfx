"""
Load Open3D Mesh
================

Demonstrates loading mesh models from open3d.
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

import open3d.visualization
from open3d.cpu.pybind.visualization.rendering import TriangleMeshModel

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
import pygfx as gfx

# load helmet model with open3d
helmet: TriangleMeshModel = io.read_triangle_model(
    str(model_dir / "DamagedHelmet/glTF/DamagedHelmet.gltf")
)

# extract helmet infos
helmet_mesh_info: TriangleMeshModel.MeshInfo = helmet.meshes[0]
helmet_mesh: open3d.geometry.TriangleMesh = helmet_mesh_info.mesh
helmet_material: open3d.visualization.Material = helmet.materials[
    helmet_mesh_info.material_idx
]

material = gfx.materials.material_from_open3d(helmet_material)

# create scene
scene = gfx.Scene()
scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

# construct helmet mesh object
mesh = gfx.Mesh(
    gfx.geometries.geometry_from_open3d(helmet_mesh),
    material,
)

# or use alternative helper method for simplified TriangleMeshModel loading
mesh2 = gfx.Group()
for m in gfx.utils.load.meshes_from_open3d(helmet):
    mesh2.add(m)

mesh2.world.position = (2, 0, 0)

scene.add(mesh)
scene.add(mesh2)

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene, up=(0, 1, 0))
