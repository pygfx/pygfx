"""
Stats helper (Manual)
=====================

Demonstrates how to display performance statistics such as FPS
and draw time on screen, by manually integrating it into
the render loop.
"""

# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

# Show something
scene = gfx.Scene()
camera = gfx.PerspectiveCamera()
box = gfx.Mesh(
    gfx.box_geometry(5, 5, 5),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)
scene.add(box)
camera.show_object(box, scale=2)

# Let there be ...
scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight(position=(0, 0, 1)))

# Add stats
stats = gfx.Stats(viewport=renderer)


def animate():
    # Rotate the cube
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    box.rotation.multiply(rot)

    # Track the render time. You can enclose any
    # other piece of code for which you want to see time
    # statistics as well.
    with stats:
        renderer.render(scene, camera, flush=False)
    stats.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
