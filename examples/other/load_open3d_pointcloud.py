"""
Load Open3D Pointcloud
======================

Demonstrates loading pointclouds from open3d.
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

import numpy as np

import pygfx as gfx

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

from importlib.util import find_spec

if not find_spec("open3d"):
    raise ImportError(
        "The `open3d` library is required for this example: pip install open3d"
    )

import open3d as o3d


def create_example_point_cloud() -> o3d.geometry.PointCloud:
    # read teapot and create pointcloud out of it
    teapot = o3d.io.read_triangle_mesh(str(model_dir / "teapot.stl"))
    pcd: o3d.geometry.PointCloud = teapot.sample_points_poisson_disk(2000)

    # give pointcloud colors
    # Assuming `point_cloud` is an open3d.geometry.PointCloud object
    points = np.asarray(pcd.points)

    # Normalize the points between 0 and 1 for RGB mapping
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    normalized_points = (points - min_bound) / (max_bound - min_bound)

    # Assign normalized points as colors
    pcd.colors = o3d.utility.Vector3dVector(normalized_points)

    # o3d.visualization.draw_geometries([pcd])
    return pcd


open3d_point_cloud = create_example_point_cloud()

# create pygfx scene
scene = gfx.Scene()
scene.add(gfx.AmbientLight(), gfx.DirectionalLight())

# create point cloud object
geometry = gfx.geometries.geometry_from_open3d(open3d_point_cloud)
material = gfx.PointsMaterial(size=2, color_mode="vertex")
points = gfx.Points(geometry, material)

scene.add(points)

if __name__ == "__main__":
    disp = gfx.Display()
    disp.show(scene, up=(0, 1, 0))
