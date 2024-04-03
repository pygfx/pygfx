"""
Simple Cube with WX
===================

Example showing a single geometric cube.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import pylinalg as la
import wx
from wgpu.gui.wx import WgpuCanvas

import pygfx as gfx

app = wx.App()

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)
scene.add(cube)

scene.add(gfx.AmbientLight())

directional_light = gfx.DirectionalLight()
directional_light.world.position = (0, 0, 1)
scene.add(directional_light)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.world.position = (0, 0, 400)


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.MainLoop()
