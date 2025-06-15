"""
Thick Lines
===========

Display very thick lines to show how lines stay pretty on large scales.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import random
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np
from pylinalg import vec_transform, vec_unproject


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)
viewport = gfx.Viewport(renderer)

# A straight line
line1 = [[100, 100], [100, 200], [100, 200], [100, 400]]

# A line with a 180 degree turn (a bit of a special case for the implementation)
line2 = [[200, 100], [200, 400], [200, 100]]

# A swiggly line
line3 = [[300 + random.randint(-10, 10), 100 + i * 3] for i in range(100)]

# A line with other turns
line4 = [[400, 100], [500, 200], [400, 300], [450, 400]]

scene = gfx.Scene()

material = gfx.LineMaterial(
    thickness=80.0,
    color=(0.8, 0.7, 0.0),
    pick_write=True,
)
line_debug = gfx.LineDebugMaterial(
    thickness=80.0,
    color=(0.8, 0.7, 0.0),
    pick_write=True,
)
points_material = gfx.PointsMaterial(
    size=10,
    color=(1.0, 0.2, 0.2),
    pick_write=True,
)

# Store all line and points objects for visibility toggling
line_objects = []
points_objects = []

for line in [line1, line2, line3, line4]:
    line = [(*pos, 0) for pos in line]  # Make the positions vec3
    geometry = gfx.Geometry(positions=line)
    line_obj = gfx.Line(geometry, material)
    scene.add(line_obj)
    line_objects.append(line_obj)

    # Add points at the vertices
    points = gfx.Points(geometry, points_material)
    points.local.z = 0.1
    points.visible = False
    scene.add(points)
    points_objects.append(points)

# State for thickness, point size, material selection, and visibility
state = {
    "thickness": material.thickness,
    "point_size": 10.0,
    "show_points": False,
    "debug_mode": False,
    "transparent": False,
}

camera = gfx.OrthographicCamera()
controller = gfx.PanZoomController(camera, register_events=renderer, damping=0)
camera.show_object(scene, match_aspect=True, scale=1.1)

# Setup ImGui renderer
try:
    from wgpu.utils.imgui import ImguiRenderer
    from imgui_bundle import imgui

    imgui_renderer = ImguiRenderer(renderer.device, canvas)
except ImportError:
    imgui_renderer = None


def draw_imgui():
    imgui.new_frame()
    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)
    is_expand, _ = imgui.begin("Controls", None)
    if is_expand:
        imgui.push_item_width(150)
        changed, state["thickness"] = imgui.slider_float(
            "Thickness", state["thickness"], 0.1, 100.0
        )
        imgui.pop_item_width()
        if changed:
            material.thickness = state["thickness"]
            line_debug.thickness = state["thickness"]

        changed, state["transparent"] = imgui.checkbox(
            "Transparent", state["transparent"]
        )
        if changed:
            alpha = 0.7 if state["transparent"] else 1.0
            current_color = material.color
            material.color = (
                current_color[0],
                current_color[1],
                current_color[2],
                alpha,
            )
            line_debug.color = (
                current_color[0],
                current_color[1],
                current_color[2],
                alpha,
            )

        imgui.same_line()
        changed, state["debug_mode"] = imgui.checkbox("Debug Mode", state["debug_mode"])
        if changed:
            mat = line_debug if state["debug_mode"] else material
            for line_obj in line_objects:
                line_obj.material = mat

        imgui.push_item_width(150)
        changed, state["point_size"] = imgui.slider_float(
            "Point Size", state["point_size"], 1.0, 100.0
        )
        imgui.pop_item_width()
        if changed:
            points_material.size = state["point_size"]

        imgui.same_line()
        changed, state["show_points"] = imgui.checkbox(
            "Show Points", state["show_points"]
        )
        if changed:
            for points in points_objects:
                points.visible = state["show_points"]
    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


if imgui_renderer is not None:
    imgui_renderer.set_gui(draw_imgui)

point_event_info = {
    "vertex_index": None,
    "world_object": None,
}


def lines_events(event):
    if event.type == "pointer_down":
        world_object = event.pick_info["world_object"]
        vertex_index = event.pick_info["vertex_index"]
        if isinstance(world_object, gfx.Line):
            clicked_on_vertex = abs(event.pick_info["segment_coord"]) <= 0.001
        elif isinstance(world_object, gfx.Points):
            clicked_on_vertex = True
        else:
            clicked_on_vertex = False
        if clicked_on_vertex:
            point_event_info["vertex_index"] = vertex_index
            point_event_info["world_object"] = world_object
            world_object.set_pointer_capture(event.pointer_id, event.root)
        return

    if point_event_info["world_object"] is None:
        return

    if event.type == "pointer_up":
        world_object = point_event_info["world_object"]
        world_object.release_pointer_capture(event.pointer_id)
        point_event_info["vertex_index"] = None
        point_event_info["world_object"] = None
        return

    vertex_index = point_event_info["vertex_index"]
    world_object = point_event_info["world_object"]

    x, y = event.x, event.y
    vs = viewport.logical_size
    # convert position to NDC
    x = x / vs[0] * 2 - 1
    y = -(y / vs[1] * 2 - 1)
    pos_ndc = (x, y, 0)

    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)

    local_coord = world_object.world.inverse_matrix @ np.asarray(
        [pos_world[0], pos_world[1], 0, 1], dtype=np.float32
    )
    world_object.geometry.positions.data[vertex_index] = local_coord[:3]
    world_object.geometry.positions.update_full()
    return


for points in points_objects:
    points.add_event_handler(
        lines_events,
        "pointer_down",
        "pointer_up",
        "pointer_move",
    )

for line in line_objects:
    line.add_event_handler(lines_events, "pointer_down", "pointer_up", "pointer_move")


def animate():
    renderer.render(scene, camera)
    imgui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
