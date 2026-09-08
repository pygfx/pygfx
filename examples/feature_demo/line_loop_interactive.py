"""
Closed shapes with material.loop = n
====================================

A field of random convex polygons, all of them corners of one buffer, drawn as
closed shapes by ``material.loop = n``: every ``n``'th node closes back to the
first node of its group. No nan-separators, no repeated end-points -- an
``(m, n, 3)`` array of corners is reshaped to ``(m * n, 3)`` and handed over as
it is.

**What to look at.** Turn the opacity down and zoom in on a corner. The shapes
are closed but nothing is drawn twice, so the seam where a shape meets itself
is exactly as dark as the rest of it. That is the whole point of ``loop``: with
a repeated end-point the first node would be painted over itself and show up as
a brighter spot. See https://github.com/pygfx/pygfx/issues/1047.

**nan loop** switches to the other way of writing the same picture: a buffer
with a nan-position after every shape and ``material.loop = True``. It should
look identical -- what differs is the cost. The nan buffer carries an extra node
per shape, and closing the shapes that way means finding the nans, which is a
scan of the positions on the CPU. Tick **positions change each frame** and watch
the bake time at the bottom of the panel: with ``loop = n`` it stays at zero,
because the shader needs no baked buffer at all.

**loop length** is both the number of corners each shape is generated with and
the value of ``material.loop``. It is the one control that rebuilds the position
buffer.

**objects** does not touch the geometry at all: the positions for the maximum
number of shapes are uploaded once, and the slider only moves
``geometry.positions.draw_range``. Ranges snap outward to whole shapes.

**dash_pattern** puts the shapes on a dashed line. Each shape restarts its
pattern from zero and its closing segment measures its own length, so the dashes
run round a shape and meet themselves rather than arriving mid-period.

Zoom with the wheel and pan by dragging. Tick **3D camera** for a perspective
view with orbit; the shapes are scattered through a slab of depth and lean at
random angles, so there is something for the perspective to do.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import math
import time

import numpy as np
import pylinalg as la
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu.shaders.lineshader import LineShader
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


# Instrumentation, for this example only -- pygfx itself carries no timing code.
# The renderer collects `shader.bake_function` from every shader that asks for
# one, so wrapping it on the class measures exactly the per-frame CPU work that
# a line's own dashing and loop buffers cost. It is a bound method that is
# looked up when a pipeline is built, so this has to be in place before the
# first render, which it is.
BAKE_WINDOW = 60  # frames the timing is averaged over
BAKE_SETTLE = 10  # frames before it is worth reporting at all
_bake_ms = [0.0]  # accumulated within the current frame
_bake_history = []
_original_bake_function = LineShader.bake_function


def _timed_bake_function(self, *args):
    t0 = time.perf_counter()
    _original_bake_function(self, *args)
    _bake_ms[0] += (time.perf_counter() - t0) * 1e3


LineShader.bake_function = _timed_bake_function


TITLE = "Closed shapes with material.loop = n"
CANVAS_SIZE = 1300, 820
PANEL_WIDTH = 340
LABEL_WIDTH = -150

SEED = 7
MAX_OBJECTS = 4000
MIN_CORNERS, MAX_CORNERS = 3, 16
FIELD = 1000.0  # half-width of the field of shapes
DEPTH = 700.0  # half-depth of the slab they are scattered through
MAX_TILT = 55.0  # degrees a shape's plane may lean away from the 2D view

DASH_PATTERNS = [
    ("none", []),
    ("2, 2", [2, 2]),
    ("4, 2", [4, 2]),
    ("1, 3", [1, 3]),
    ("6, 2, 2, 2", [6, 2, 2, 2]),
]


def make_positions(corners):
    """The corners of MAX_OBJECTS random convex polygons, as (MAX_OBJECTS * n, 3).

    Convex by construction: the corners of each shape sit on a circle at
    increasing angles, which is always a convex cyclic polygon, and the scale,
    tilt and translation applied afterwards are affine, so they keep it convex.

    The per-object draws come first and in a fixed order, so changing `corners`
    re-rolls the shapes but leaves every centre, size and tilt where it was.
    """

    def normalized(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    rng = np.random.default_rng(SEED)
    centers = rng.uniform(-1, 1, (MAX_OBJECTS, 1, 3)) * (FIELD, FIELD, DEPTH)
    # Two half-axes, unequal so the shapes are not all near-circular
    rx = rng.uniform(12.0, 55.0, (MAX_OBJECTS, 1, 1))
    ry = rx * rng.uniform(0.35, 1.0, (MAX_OBJECTS, 1, 1))
    # A random plane per shape, so that the 3D view has something to show. Its
    # normal is drawn from a cone about the 2D view direction rather than from
    # the whole sphere, so that face-on the shapes still read as shapes instead
    # of collapsing to slivers; orbiting is what turns them edge-on.
    lean = np.arccos(
        rng.uniform(math.cos(math.radians(MAX_TILT)), 1.0, (MAX_OBJECTS, 1))
    )
    around = rng.uniform(0, 2 * np.pi, (MAX_OBJECTS, 1))
    normal = np.stack(
        [np.sin(lean) * np.cos(around), np.sin(lean) * np.sin(around), np.cos(lean)],
        axis=-1,
    )  # (m, 1, 3)
    # Any two perpendicular axes within that plane; the in-plane orientation is
    # random because the vector crossed in is.
    u = normalized(np.cross(normal, rng.normal(size=(MAX_OBJECTS, 1, 3))))
    v = np.cross(normal, u)

    # Evenly spaced angles with a bounded jitter: varied shapes, but no two
    # corners so close together that the join degenerates.
    steps = np.arange(corners) + rng.uniform(-0.3, 0.3, (MAX_OBJECTS, corners))
    angles = (steps * (2 * np.pi / corners))[..., np.newaxis]  # (m, n, 1)

    positions = centers + rx * np.cos(angles) * u + ry * np.sin(angles) * v
    return positions.reshape(-1, 3).astype(np.float32)


def with_nans(positions, corners):
    """The same shapes, written the classic way: a nan-position after each one.

    That is what `material.loop = True` needs in order to tell the shapes apart,
    and it is what `material.loop = n` exists to avoid: an extra node per shape,
    and a copy of the whole array to put it there.
    """
    shapes = positions.reshape(-1, corners, 3)
    out = np.full((len(shapes), corners + 1, 3), np.nan, np.float32)
    out[:, :corners] = shapes
    return out.reshape(-1, 3)


_geometries = {}


def get_geometry(corners, nan):
    """The geometry for this corner count, in either layout, built once.

    Only the current corner count is kept; both layouts of it are built together
    so that toggling `nan loop` does not rebuild anything.
    """
    if (corners, nan) not in _geometries:
        _geometries.clear()
        positions = make_positions(corners)
        _geometries[corners, False] = gfx.Geometry(positions=positions)
        _geometries[corners, True] = gfx.Geometry(
            positions=with_nans(positions, corners)
        )
    return _geometries[corners, nan]


canvas = RenderCanvas(size=CANVAS_SIZE, title=TITLE)
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

material = gfx.LineMaterial(
    thickness=6,
    color="#4cf",
    alpha_mode="blend",
    opacity=0.45,
    loop=4,
    aa=True,
    thickness_space="screen",
)
line = gfx.Line(get_geometry(4, False), material)
scene.add(line)

# An orthographic camera in pygfx *is* a perspective one with fov 0 -- they are
# the same class -- so the 2D and 3D views here are one camera with two values
# of `fov`, and switching between them keeps the position, orientation and view
# extent untouched. What changes is the projection, and which controller drives
# it.
VIEW_SIZE = CANVAS_SIZE[0] - PANEL_WIDTH, CANVAS_SIZE[1]
camera = gfx.PerspectiveCamera(0)
camera.width = 2.4 * FIELD
camera.height = camera.width * VIEW_SIZE[1] / VIEW_SIZE[0]

# The controller measures pan, zoom and rotate against a viewport's rect, so it
# is given a viewport covering the render area rather than the whole canvas.
# That is also what keeps the panel usable: `pointer_down` and `wheel` are gated
# on `Viewport.is_inside`, so events over the panel fall outside and dragging a
# slider does not move the camera as well.
view = gfx.Viewport(renderer, rect=(0, 0, *VIEW_SIZE))
panzoom = gfx.PanZoomController(camera)
orbit = gfx.OrbitController(camera)


def active_controller():
    return orbit if state["camera_3d"] else panzoom


def standoff(fov):
    """How far back the camera must sit for `fov` to span the current extent.

    Zero for an orthographic camera, which has no such distance: its rays are
    parallel, so it can sit in the plane it is looking at -- which is where this
    one starts, and why switching to a perspective projection without moving it
    would put the scene at w = 0.

    pygfx builds both projections from the same reference size, the mean of the
    camera's width and height, and applies the same aspect correction to each,
    so that cancels and equating the two extents at distance D gives this. The
    zoom factor cancels as well, so a zoomed view switches just as cleanly.
    """
    if fov <= 0:
        return 0.0
    return 0.25 * (camera.width + camera.height) / math.tan(math.radians(fov) / 2)


def set_fov(fov):
    """Change the projection while keeping the framing.

    The camera is dollied along its own axis by the change in standoff, so the
    plane it was looking at still spans the same extent afterwards. Nothing on
    screen moves; only the perspective does.
    """
    if fov == camera.fov:
        return
    delta = standoff(fov) - standoff(camera.fov)
    camera.local.position = camera.local.position + la.vec_transform_quat(
        (0, 0, delta), camera.local.rotation
    )
    camera.fov = fov


renderer.add_event_handler(
    lambda event: active_controller().handle_event(event, view),
    "pointer_down",
    "pointer_move",
    "pointer_up",
    "key_down",
    "key_up",
    "wheel",
    "before_render",
)
home = camera.get_state()

gui_renderer = ImguiRenderer(renderer.device, canvas)

state = {
    "opacity": 0.45,
    "thickness": 6.0,
    "loop_length": 4,
    "n_objects": 400,
    "nan_loop": False,
    "stream": False,
    "dash_pattern": 0,
    "dash_offset": 0.0,
    # Tick for a perspective camera with orbit; see `set_fov`.
    "camera_3d": False,
    "fov": 45.0,
}
OPENING_STATE = dict(state)
applied = {}


def apply_state():
    if state == applied:
        return
    corners = state["loop_length"]
    nan = state["nan_loop"]
    applied.clear()
    applied.update(state)
    # The timing below is a rolling mean, so it has to start over whenever a
    # control moves; otherwise it reports the previous setting for a second.
    _bake_history.clear()

    # The two layouts of the same shapes. Swapping between them is free; a new
    # corner count is the one thing that rebuilds a buffer.
    geometry = get_geometry(corners, nan)
    if line.geometry is not geometry:
        line.geometry = geometry

    # The object count is only a draw range: the positions for MAX_OBJECTS are
    # already on the GPU, and nothing is copied, re-baked or re-uploaded.
    stride = corners + 1 if nan else corners
    line.geometry.positions.draw_range = 0, state["n_objects"] * stride

    material.opacity = state["opacity"]
    material.thickness = state["thickness"]
    material.loop = True if nan else corners
    material.dash_pattern = DASH_PATTERNS[state["dash_pattern"]][1]
    material.dash_offset = state["dash_offset"]

    set_fov(state["fov"] if state["camera_3d"] else 0.0)


def draw_imgui():
    display = gui_renderer.backend.io.display_size
    imgui.set_next_window_size((PANEL_WIDTH, display.y), imgui.Cond_.always)
    imgui.set_next_window_pos((display.x - PANEL_WIDTH, 0), imgui.Cond_.always)
    is_expand, _ = imgui.begin(
        TITLE,
        None,
        flags=imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_collapse,
    )
    if is_expand:
        imgui.push_item_width(LABEL_WIDTH)

        if imgui.button("reset"):
            state.update(OPENING_STATE)
        imgui.same_line()
        if imgui.button("reset view"):
            camera.set_state(home)
            set_fov(state["fov"] if state["camera_3d"] else 0.0)

        imgui.separator_text("The shapes")
        _, state["n_objects"] = imgui.slider_int(
            "objects", state["n_objects"], 1, MAX_OBJECTS
        )
        imgui.set_item_tooltip(
            "Only moves geometry.positions.draw_range. The corners of all\n"
            f"{MAX_OBJECTS} shapes are uploaded once and never touched again,\n"
            "so this costs nothing but the extra vertices.\n\n"
            "A range that does not start or end on a shape boundary snaps\n"
            "outward to one; an incomplete trailing shape is not drawn."
        )
        _, state["loop_length"] = imgui.slider_int(
            "loop length", state["loop_length"], MIN_CORNERS, MAX_CORNERS
        )
        imgui.set_item_tooltip(
            "material.loop -- how many nodes the shader treats as one closed\n"
            "shape, and equally how many corners each shape is generated\n"
            "with. The one control that rebuilds the position buffer."
        )

        imgui.separator_text("The line")
        _, state["opacity"] = imgui.slider_float("opacity", state["opacity"], 0.05, 1.0)
        imgui.set_item_tooltip(
            "Turn this down and zoom in on a shape. Every part of it is the\n"
            "same darkness, seam included -- nothing is drawn twice."
        )
        _, state["thickness"] = imgui.slider_float(
            "thickness", state["thickness"], 0.5, 40.0
        )
        _, state["nan_loop"] = imgui.checkbox("nan loop", state["nan_loop"])
        imgui.set_item_tooltip(
            "Draw the same shapes the classic way: a buffer with a nan-position\n"
            "after each shape and material.loop = True.\n\n"
            "The picture is the same. What differs is one extra node per shape,\n"
            "and a scan of the positions to find the nans whenever they change\n"
            "-- which is what the bake time below measures."
        )

        imgui.separator_text("Dashing")
        _, state["dash_pattern"] = imgui.combo(
            "dash_pattern",
            state["dash_pattern"],
            [name for name, _ in DASH_PATTERNS],
            len(DASH_PATTERNS),
        )
        imgui.set_item_tooltip(
            "Dashing needs a cumulative distance per node, which is baked on\n"
            "the CPU. With loop=n it is baked in the shader's virtual node\n"
            "space, one extra slot per shape, so each shape's dashes start\n"
            "at zero and its closing segment measures its own length rather\n"
            "than the whole perimeter.\n\n"
            "With 'none' there is no bake at all: loop=n is then pure index\n"
            "arithmetic in the vertex shader, and nothing runs per frame."
        )
        imgui.begin_disabled(not DASH_PATTERNS[state["dash_pattern"]][1])
        _, state["dash_offset"] = imgui.slider_float(
            "dash_offset", state["dash_offset"], 0.0, 4.0
        )
        imgui.set_item_tooltip(
            "The phase to start each shape's pattern at. Every shape starts\n"
            "from the same phase, because each one restarts its cumulative\n"
            "distance at zero."
        )
        imgui.end_disabled()

        imgui.separator_text("Camera")
        _, state["camera_3d"] = imgui.checkbox("3D camera", state["camera_3d"])
        imgui.set_item_tooltip(
            "Switch between an orthographic camera with pan/zoom and a\n"
            "perspective one with orbit, in place. Position, orientation and\n"
            "framing are kept, so nothing changes size or moves off screen.\n\n"
            "The shapes are scattered through a slab of depth and tilted at\n"
            "random, so orbiting shows them edge-on as well as face-on."
        )
        imgui.begin_disabled(not state["camera_3d"])
        _, state["fov"] = imgui.slider_float("fov", state["fov"], 5.0, 120.0)
        imgui.end_disabled()

        imgui.separator_text("Cost")
        _, state["stream"] = imgui.checkbox(
            "positions change each frame", state["stream"]
        )
        imgui.set_item_tooltip(
            "Mark the position buffer dirty every frame, as an app streaming\n"
            "new positions would.\n\n"
            "The loop bake is cached against the buffer's revision, so this is\n"
            "what makes it run. (A screen-space dash bake depends on the camera\n"
            "instead, so that one runs every frame either way.)"
        )
        positions = line.geometry.positions
        drawn = state["n_objects"] * (
            state["loop_length"] + 1 if state["nan_loop"] else state["loop_length"]
        )
        imgui.text(f"drawn: {drawn} of {positions.nitems} nodes")
        imgui.text(f"buffer: {positions.nbytes / 1e6:.2f} MB")
        if len(_bake_history) < BAKE_SETTLE:
            imgui.text("bake: measuring...")
        else:
            mean_ms = sum(_bake_history) / len(_bake_history)
            imgui.text(f"bake: {mean_ms:.3f} ms/frame")
            if mean_ms == 0.0:
                imgui.text_wrapped(
                    "Nothing to bake. Undashed, loop = n is pure index "
                    "arithmetic in the vertex shader: no buffer, and no CPU "
                    "work per frame however the positions change."
                    if not state["nan_loop"]
                    else "Nothing to bake -- the positions have not changed "
                    "since the last one. Tick the box above."
                )

        imgui.pop_item_width()
    imgui.end()


gui_renderer.set_gui(draw_imgui)


def animate():
    apply_state()
    if state["stream"]:
        # What an app that moves its data every frame costs. The data does not
        # actually change here; marking the buffer dirty is what re-triggers the
        # upload and the bake, which is the part being measured.
        line.geometry.positions.update_full()

    width, height = canvas.get_logical_size()
    view.rect = 0, 0, width - PANEL_WIDTH, height
    _bake_ms[0] = 0.0
    renderer.render(scene, camera, flush=False, rect=view.rect)
    renderer.flush()

    _bake_history.append(_bake_ms[0])
    del _bake_history[:-BAKE_WINDOW]

    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    renderer.request_draw(animate)
    loop.run()
