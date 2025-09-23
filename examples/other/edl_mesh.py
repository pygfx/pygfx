"""
Stanford Bunny mesh with Eye-Dome Lighting (EDL)
================================================

This example loads the Stanford Bunny mesh via Open3D, converts it to pygfx,
renders with a basic material, and applies EDL as a post-process.
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


def load_open3d_bunny_mesh() -> o3d.geometry.TriangleMesh:
    mesh_path = o3d.data.BunnyMesh().path
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("Failed to load Bunny mesh via Open3D data")
    return mesh


canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#111"))

o3d_mesh = load_open3d_bunny_mesh()
geometry = gfx.geometries.geometry_from_open3d(o3d_mesh)
material = gfx.MeshBasicMaterial(color="#e6e6e6")
mesh = gfx.Mesh(geometry, material)
scene.add(mesh)

camera = gfx.PerspectiveCamera(50, 1)
camera.show_object(mesh, view_dir=(1, -1, 0.6))

controller = gfx.OrbitController(camera, register_events=renderer)

scene.add(gfx.AmbientLight(0.2), camera.add(gfx.DirectionalLight(1.0)))

renderer.effect_passes = [EDLPass(strength=10.0, radius=1.5, depth_edge_threshold=0.0)]


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
