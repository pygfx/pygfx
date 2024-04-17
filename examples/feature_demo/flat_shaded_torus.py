"""
Flat Shading
============

Example showing a Torus knot, using flat shading.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


geometry = gfx.torus_knot_geometry(1, 0.3, 64, 10)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshPhongMaterial(color=(1, 0, 1, 1), flat_shading=True)
obj = gfx.Mesh(geometry, material)
scene.add(obj)

camera = gfx.PerspectiveCamera(70, 1)
camera.local.z = 4
scene.add(camera.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
