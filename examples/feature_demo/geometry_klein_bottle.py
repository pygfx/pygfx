"""
Klein Bottle Geometry
=====================

Example showing a Klein Bottle.
"""

# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

geometry = gfx.klein_bottle_geometry(10)
geometry.texcoords = None
material = gfx.MeshPhongMaterial(color=(1, 0.5, 0, 1))
obj = gfx.Mesh(geometry, material)
scene.add(obj)

obj2 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color="#00f", line_length=1))
obj3 = gfx.Mesh(geometry, gfx.MeshNormalLinesMaterial(color="#0ff", line_length=-1))
obj.add(obj2, obj3)

camera = gfx.PerspectiveCamera(70, 1)
camera.position.z = 30

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))


def animate():
    # rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.0071, 0.01))
    # obj.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
