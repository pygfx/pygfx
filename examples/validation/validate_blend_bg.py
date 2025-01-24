"""
Validate transparency with and without bg
=========================================

Shows a semi-transparent square. On the left it uses a background, on
the right it does not.

In the latter case, there is no background to blend with, so the
resulting pixels are still semi-transparent. When these final pixels
are shown on screen, they are blended with a virtual black background
(i.e. the alpha channel is applied), resulting in a darker appearance.

The semi-transparent regions may also be blended with your desktop, if
the OS and GUI library are configured to do so.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.blend_mode = "ordered2"

geometry = gfx.plane_geometry(100, 100)

plane_bg = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 0, 1)))
plane_red = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.3)))

plane_bg.local.position = -50, 0, 0
plane_red.local.position = 0, 0, 10

scene = gfx.Scene()
scene.add(plane_bg, plane_red)

camera = gfx.OrthographicCamera(100, 100)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
