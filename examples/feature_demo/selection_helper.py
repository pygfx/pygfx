"""
Selection Helper
=================

Example demonstration how to use the selection helper
to select data points:
 - shift-click and drag the mouse to select points
 - shift-control-click (shift-command-click on macs) to add points to the selection
 - double click to deselect all points

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

xx = np.random.rand(10)
yy = np.random.rand(10)

colors = np.ones((10, 4), dtype=np.float32)
colors[:, 0] = 0

geometry = gfx.Geometry(positions=[(x, y, 0) for x, y in zip(xx, yy)], colors=colors)
ob = gfx.Points(
    geometry, gfx.PointsMaterial(size=20, pick_write=True, color_mode="vertex")
)
scene.add(ob)

camera = gfx.OrthographicCamera(120, 120)
camera.show_object(ob)

selected = np.zeros(len(xx), dtype=bool)


def update_selection(sel):
    """Highlight selected points."""
    # Get the world coordinates of the points
    pos_world = la.vec_transform(ob.geometry.positions.data, ob.world.matrix)

    # Check if the points are inside the selection box
    inside = sel.is_inside(pos_world)

    # Update selection
    select(inside, additive="Control" in sel._event_modifiers)


def select(new_selected, additive=False):
    """Select given points."""
    # See if we actually need to change any colors
    if any(selected != new_selected):
        if additive:
            selected[:] = selected | new_selected
        else:
            selected[:] = new_selected
        ob.geometry.colors.data[:, :] = [0, 1, 1, 1]  # reset to cyan
        if any(selected):
            ob.geometry.colors.data[selected, :] = [1, 1, 0, 1]  # set to yellow
        ob.geometry.colors.update_range()

        # We need to request a redraw to make sure the colors are updated
        canvas.request_draw(lambda: renderer.render(scene, camera))


def deselect_all():
    """Deselect all points."""
    select(np.zeros(len(xx), dtype=bool))


selection = gfx.SelectionGizmo(
    renderer=renderer,
    camera=camera,
    scene=scene,
    show_info=True,  # set to false to hide info text
    debug=False,  # set to true to get debug output
    callback_during_drag=update_selection,
    callback_after_drag=update_selection,
)

renderer.add_event_handler(lambda x: deselect_all(), "double_click")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
