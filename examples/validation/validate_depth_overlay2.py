"""
Depth Overlay 2
===============

Example (and test) for behavior of the depth buffer w.r.t. overlays,
implemented using ``Material.depth_test``. The overlaid object should
always be on top.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


# Create a canvas and renderer

canvas = WgpuCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# Compose a scene with a 3D cube at the origin

# Define cubes in the order that we want them rendered
cubes = [
    (2.0, "#f00", 1),
    (1.8, "#ff0", 1),
    (1.6, "#0f0", 2),
    (1.4, "#0ff", 2),
]

# We can mangle them a bit, because we define render order
cubes = cubes[2:] + cubes[0:2]

for size, color, render_order in cubes:
    cube = gfx.Mesh(
        gfx.box_geometry(size, size, size),
        gfx.MeshPhongMaterial(color=color, side="front", depth_test=False),
    )
    cube.render_order = render_order
    rot = la.quat_from_euler((0.2, 0.3), order="XY")
    cube.local.rotation = la.quat_mul(rot, cube.local.rotation)
    scene.add(cube)


# Camera

camera = gfx.OrthographicCamera(3, 3)
scene.add(camera.add(gfx.DirectionalLight()))

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
