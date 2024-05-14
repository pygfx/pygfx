"""
Validate the grid
=================

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()

grid = gfx.Grid(None, gfx.GridMaterial())
scene.add(grid)

box = gfx.Mesh(
    gfx.box_geometry(),
    gfx.MeshPhongMaterial(color="green"),
)
scene.add(box)

# camera = gfx.OrthographicCamera(5, 5)
camera = gfx.PerspectiveCamera()
camera.local.position = (5, 5, 5)
camera.show_object(box)

scene.add(camera.add(gfx.DirectionalLight()))

controller = gfx.OrbitController(camera, register_events=renderer)
# controller = gfx.FlyController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
