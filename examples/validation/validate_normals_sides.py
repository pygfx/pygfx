"""
Normals on front and back
=========================

This example validates that normals are handled such that both the front
and back faces are lit. Further it validates the appearance of the
normal lines material.

This shows a flattened cylinder, with normal lines sticking out both
at the front and back faces.
"""
# test_example = true
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

geo = gfx.cylinder_geometry(radial_segments=16, height_segments=3, open_ended=True)
ob1 = gfx.Mesh(
    geo,
    gfx.MeshPhongMaterial(color="#f00"),
)
ob2 = gfx.Mesh(
    geo,
    gfx.MeshNormalLinesMaterial(color="#0f0", line_length=0.4),
)
ob3 = gfx.Mesh(
    geo,
    gfx.MeshNormalLinesMaterial(color="#00f", line_length=-0.2),
)

ob1.position.set(1, 0, 0)
ob1.rotation.set_from_axis_angle(gfx.linalg.Vector3(0, 1, 0), 1)
ob1.scale.set(3, 1, 1)
scene.add(ob1.add(ob2, ob3))

scene.add(gfx.DirectionalLight(), gfx.AmbientLight())

camera = gfx.PerspectiveCamera(70, 1, depth_range=(0.1, 2000))
camera.position.z = 4.5

renderer.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
