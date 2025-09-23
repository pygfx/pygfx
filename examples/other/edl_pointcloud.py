"""
Point cloud with Eye-Dome Lighting (EDL)
========================================

This example renders the Stanford Bunny point cloud (via Open3D) and applies
EDL as a post-process to enhance depth perception.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

from importlib.util import find_spec

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu.engine.edl import EDLPass

if not find_spec("open3d"):
    raise ImportError(
        "The `open3d` library is required for this example: pip install open3d"
    )

import open3d as o3d


def load_open3d_bunny_pointcloud(max_points: int = 40000) -> o3d.geometry.PointCloud:
    # Prefer the Open3D bunny point cloud dataset if available
    try:
        pcd_path = o3d.data.BunnyPointCloud().path
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) > 0:
            # Uniformly downsample if too many points
            if len(pcd.points) > max_points:
                stride = max(1, len(pcd.points) // max_points)
                pcd = pcd.uniform_down_sample(stride)
            return pcd
    except Exception:
        pass

    # Fallback: load bunny mesh dataset and sample points
    mesh_path = o3d.data.BunnyMesh().path
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices():
        raise RuntimeError("Failed to load Bunny mesh via Open3D data")
    # Poisson disk sampling for a clean distribution
    pcd = mesh.sample_points_poisson_disk(max_points)
    return pcd


canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Background for contrast
scene.add(gfx.Background.from_color("#111"))

# Load Stanford Bunny with Open3D and render as point cloud
pcd = load_open3d_bunny_pointcloud(max_points=40000)
geometry = gfx.geometries.geometry_from_open3d(pcd)
material = gfx.PointsMaterial(size=8.0, color="#ddd", aa=False, size_space="screen")
points = gfx.Points(geometry, material)
scene.add(points)

camera = gfx.PerspectiveCamera(60, 1)
camera.show_object(scene, view_dir=(1, -1, 0.8))

controller = gfx.OrbitController(camera, register_events=renderer)

# Add lights (not strictly needed for points, but harmless)
scene.add(gfx.AmbientLight(0.4), camera.add(gfx.DirectionalLight(0.8)))

# Apply EDL as post-processing
renderer.effect_passes = [EDLPass(strength=10.0, radius=1.5, depth_edge_threshold=0.0)]


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
