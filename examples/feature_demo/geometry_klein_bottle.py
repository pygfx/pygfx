"""
Klein Bottle Geometry
=====================

Example showing a Klein Bottle. Surface normals are shown on both sides
of the mesh, in different colors. It can be seen how the object turns itself
"inside out".

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.klein_bottle_geometry(10, stitch=False)
geometry.texcoords = None
material = gfx.MeshPhongMaterial(color=(1, 0.5, 0, 1), flat_shading=True)
obj = gfx.Mesh(geometry, material)
scene.add(obj)

obj2 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color="#00f", line_length=1))
obj3 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color="#0ff", line_length=-1))
obj.add(obj2, obj3)

camera = gfx.PerspectiveCamera(70, 1)
camera.local.z = 30

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.0071, 0.01), order="XY")
    obj.local.rotation = la.quat_mul(rot, obj.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
