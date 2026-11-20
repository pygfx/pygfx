"""
Dashing in 3D, side by side
===========================

The dashing controls of ``line_dashing_interactive.py``, but on a perspective
camera and as a split view: **left is pygfx's current behaviour, right is
whatever the panel says**. One camera drives both halves, so they are always
looking at the same thing from the same place and the only difference is the
material.

The scene is the wireframe-cubes scene from
``validation/validate_line_thickness_space.py`` -- three cubes of the same
apparent size, but built at model scales of 1, 1/2 and 1/3.3 inside containers
scaled to match. That is what makes ``thickness_space`` interesting: in model
space the three are drawn at three different dash sizes even though they look
the same size on screen.

Each cube is one ``Line``: two square loops joined by four uprights, as
nan-separated pieces. ``LineSegmentMaterial`` is deliberately not used, because
it computes its cumulative distance in the shader rather than in the bake, so
``dash_scaling`` and ``dash_fit`` do not apply to it.

**What 3D adds.** Everything the 2D example shows still holds. The extra
question here is what one dash unit is *worth* on screen, which stops being one
number the moment a line has any extent along the view direction: a segment
running away from the camera covers far fewer pixels per model unit than one
across the view.

The dash scale is therefore taken per segment. Press **continuous**
(``dash_fit=exact``, ``dash_scaling=quantized``, ``dash_scale_step=1``) and the
two halves agree: exactly at ``fov=0``, and to within a fraction of a dash under
perspective. Turn a cube edge-on with quantization on and its receding edges keep
roughly the dash size the pattern asks for, instead of being crammed with the
same number of dashes as the edges facing you.

What perspective still costs, and it is much smaller: the phase is linear in
model space, while the projection is not, so within one strongly foreshortened
segment the dashes bunch slightly toward the far end. At the nodes the two
halves match to about 1e-6; between them the ink coverage matches to 0.2% and
the dashes sit a fraction of a period out. Removing that too would need the
scale chosen per fragment rather than per segment.

The flat square floating above the cubes is a useful reference for the opposite
reason: it has no extent along the view direction when you face it, so it is the
one shape for which nothing subtle is happening at all.

Orbit with the left mouse button and zoom with the wheel, in **either** pane;
one camera drives both, so the halves always show the same view.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import math

import numpy as np
import pylinalg as la
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


TITLE = "Dashing in 3D, side by side"
CANVAS_SIZE = 1500, 820
PANEL_WIDTH = 360
LABEL_WIDTH = -170

DASH_PATTERNS = [
    ("2, 2", [2, 2]),
    ("4, 2", [4, 2]),
    ("1, 3", [1, 3]),
    ("6, 2, 2, 2", [6, 2, 2, 2]),
]
THICKNESS_SPACES = ["screen", "world", "model"]
DASH_SCALINGS = ["continuous", "quantized"]
DASH_FITS = ["exact", "stretch", "spread"]
PPAA_MODES = ["none", "fxaa", "ddaa"]
PIXEL_FILTERS = ["nearest", "linear", "tent", "disk", "bspline", "mitchell", "catmull"]
PIXEL_RATIOS = [1.0, 2.0]


def cube_pieces(size=50.0):
    """A cube as nan-separated pieces: two square loops and four uprights.

    Drawn this way rather than as twelve loose segments so that there are real
    joins and closed pieces to dash, which is where the pattern's behaviour
    actually shows.
    """
    s = size
    corners = [(-s, -s), (s, -s), (s, s), (-s, s)]
    bottom = np.array([[x, y, -s] for x, y in corners], np.float32)
    top = np.array([[x, y, s] for x, y in corners], np.float32)
    uprights = [np.array([[x, y, -s], [x, y, s]], np.float32) for x, y in corners]
    return [bottom, top, *uprights]


def joined(parts):
    nanpoint = np.full((1, 3), np.nan, np.float32)
    stacked = [parts[0]]
    for part in parts[1:]:
        stacked += [nanpoint, part]
    return np.vstack(stacked).astype(np.float32)


CUBE = joined(cube_pieces())

# A flat square, as the control: it has no extent along the view direction when
# you look straight at it, which is the one case where a single per-object scale
# factor is exactly right.
_q = 60.0
PANEL_SQUARE = np.array(
    [[-_q, -_q, 0], [_q, -_q, 0], [_q, _q, 0], [-_q, _q, 0]], np.float32
)


def build_side(color):
    """One scene: three cubes of equal apparent size but different model scale.

    The middle and right cubes are built small and placed in containers scaled
    back up, so that model space and world space disagree with each other and
    with screen space. Straight from validate_line_thickness_space.py.
    """
    material = gfx.LineMaterial(
        thickness=8,
        color=color,
        loop=True,
        aa=True,
        dash_pattern=[2, 2],
        thickness_space="screen",
    )
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    # A row, as in the original scene. The view direction below is mostly
    # front-on so that the three stay side by side rather than stacking up.
    for divisor, x in ((1.0, -210.0), (2.0, 0.0), (3.3333, 210.0)):
        container = gfx.Group()
        container.add(gfx.Line(gfx.Geometry(positions=CUBE / divisor), material))
        container.local.scale = divisor
        # The container's position is in parent space, so it is not scaled by
        # the container's own scale; it is x, not x / divisor.
        container.local.position = (x, 0, 0)
        scene.add(container)

    flat = gfx.Line(gfx.Geometry(positions=PANEL_SQUARE), material)
    flat.local.position = (0, 175, 0)
    scene.add(flat)
    return scene, material


canvas = RenderCanvas(size=CANVAS_SIZE, title=TITLE)
renderer = gfx.WgpuRenderer(canvas)

scene_left, control = build_side("#7af")
scene_right, live = build_side("#fd5")

# The camera is framed for one half-width viewport, not for the whole canvas,
# so that show_object fits the scene into the pane it will actually appear in.
VIEW_ASPECT = (CANVAS_SIZE[0] - PANEL_WIDTH) / 2 / CANVAS_SIZE[1]
camera = gfx.PerspectiveCamera(35, VIEW_ASPECT)
camera.show_object(scene_left, view_dir=(-0.3, -0.35, -1.0), scale=1.25)

# One viewport per pane, and the controller is handed whichever one the pointer
# is over. It measures its gestures against the viewport's rect, so that rect has
# to be the pane actually under the cursor: registered on the renderer it would
# scale by the whole canvas width, and registered on one pane the other half
# would be dead, because `pointer_down` and `wheel` are gated on `is_inside`.
#
# That gating is also what keeps the panel usable: events over it fall in neither
# rect, so dragging a slider does not orbit the camera as well.
VIEW_W = (CANVAS_SIZE[0] - PANEL_WIDTH) / 2
view_left = gfx.Viewport(renderer, rect=(0, 0, VIEW_W, CANVAS_SIZE[1]))
view_right = gfx.Viewport(renderer, rect=(VIEW_W, 0, VIEW_W, CANVAS_SIZE[1]))
panzoom = gfx.PanZoomController(camera)
orbit = gfx.OrbitController(camera)


def active_controller():
    return orbit if state["camera_3d"] else panzoom


def standoff(fov):
    """How far back the camera must sit for `fov` to span the current extent.

    Zero for an orthographic camera, whose rays are parallel and which
    therefore has no such distance. pygfx builds both projections from the same
    reference size -- the mean of the camera's width and height -- and applies
    the same aspect correction to each, so that cancels and equating the two
    extents at distance D gives this. The zoom factor cancels too.

    It is also exactly where `show_object` puts the camera, checked: for this
    scene it returns 1192.0 against a measured 1192.0.
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


def route_to_pane(event):
    """Give the event to the pane the pointer is in."""
    x = getattr(event, "x", -1)
    pane = view_right if x >= view_left.rect[2] else view_left
    active_controller().handle_event(event, pane)


renderer.add_event_handler(
    route_to_pane,
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
    "dash_fit": DASH_FITS.index("spread"),
    "dash_scaling": 1,
    "dash_scale_step": 2.0,
    "centre_max_scale": True,
    "dash_scale_hysteresis": 0.1,
    "dash_max_scale": 1.0,
    "pattern": 0,
    "dash_offset": 0.0,
    "thickness": 8.0,
    "thickness_space": 0,
    "loop": True,
    "aa": True,
    # Untick for an orthographic camera with pan/zoom; see `set_fov`.
    "camera_3d": True,
    "fov": 35.0,
    "ppaa": PPAA_MODES.index(renderer.ppaa),
    "pixel_filter": PIXEL_FILTERS.index(renderer.pixel_filter),
    "pixel_ratio": min(
        range(len(PIXEL_RATIOS)),
        key=lambda i: abs(PIXEL_RATIOS[i] - renderer.pixel_ratio),
    ),
}
OPENING_STATE = dict(state)
# The point of the family where quantization does nothing, i.e. what pygfx does
# on main. Under a perspective camera this still differs from the left half,
# which is the thing this example exists to show; set fov to 0 to see them meet.
CONTINUOUS_STATE = OPENING_STATE | {
    "dash_fit": DASH_FITS.index("exact"),
    "dash_scaling": DASH_SCALINGS.index("quantized"),
    "dash_scale_step": 1.0,
}
applied = {}


def apply_state():
    """Push the panel onto the right-hand material, and the shared bits onto both."""
    if state == applied:
        return
    applied.clear()
    applied.update(state)

    _, pattern = DASH_PATTERNS[state["pattern"]]
    for material in (live, control):
        material.dash_pattern = pattern
        material.dash_offset = state["dash_offset"]
        material.thickness = state["thickness"]
        material.thickness_space = THICKNESS_SPACES[state["thickness_space"]]
        material.loop = state["loop"]
        material.aa = state["aa"]

    live.dash_fit = DASH_FITS[state["dash_fit"]]
    live.dash_scaling = DASH_SCALINGS[state["dash_scaling"]]
    live.dash_scale_step = state["dash_scale_step"]
    live.dash_max_scale = None if state["centre_max_scale"] else state["dash_max_scale"]
    live.dash_scale_hysteresis = state["dash_scale_hysteresis"]

    set_fov(state["fov"] if state["camera_3d"] else 0.0)
    renderer.ppaa = PPAA_MODES[state["ppaa"]]
    renderer.pixel_filter = PIXEL_FILTERS[state["pixel_filter"]]
    renderer.pixel_ratio = PIXEL_RATIOS[state["pixel_ratio"]]


def draw_overlay():
    """The two captions and the divider, drawn over the viewports."""
    view_w = (gui_renderer.backend.io.display_size.x - PANEL_WIDTH) / 2
    height = gui_renderer.backend.io.display_size.y
    draw = imgui.get_foreground_draw_list()
    draw.add_line(
        imgui.ImVec2(view_w, 0),
        imgui.ImVec2(view_w, height),
        imgui.IM_COL32(90, 90, 90, 255),
    )
    draw.add_text(
        imgui.ImVec2(16, 14),
        imgui.IM_COL32(120, 170, 255, 255),
        "left: pygfx defaults",
    )
    draw.add_text(
        imgui.ImVec2(view_w + 16, 14),
        imgui.IM_COL32(255, 221, 85, 255),
        "right: driven by the panel",
    )


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
        imgui.set_item_tooltip(
            "Put every control back to how the example opened.\n"
            "Does not move the camera; use 'reset view' for that."
        )
        imgui.same_line()
        if imgui.button("continuous"):
            state.update(CONTINUOUS_STATE)
        imgui.set_item_tooltip(
            "dash_fit=exact, dash_scaling=quantized, dash_scale_step=1 --\n"
            "the point of the family where quantization does nothing.\n\n"
            "Makes the two halves agree: exactly at fov=0, and to within a\n"
            "fraction of a dash under perspective. The scale is taken per\n"
            "segment, so a receding edge keeps the dash size asked for."
        )

        imgui.separator_text("Fitting (right half only)")
        _, state["dash_fit"] = imgui.combo(
            "dash_fit", state["dash_fit"], DASH_FITS, len(DASH_FITS)
        )

        imgui.separator_text("Quantization (right half only)")
        _, state["dash_scaling"] = imgui.combo(
            "dash_scaling", state["dash_scaling"], DASH_SCALINGS, len(DASH_SCALINGS)
        )
        quantized = DASH_SCALINGS[state["dash_scaling"]] == "quantized"
        imgui.begin_disabled(not quantized)
        _, state["dash_scale_step"] = imgui.slider_float(
            "dash_scale_step", state["dash_scale_step"], 1.0, 4.0
        )
        _, state["centre_max_scale"] = imgui.checkbox(
            "dash_max_scale = None (centred)", state["centre_max_scale"]
        )
        imgui.begin_disabled(state["centre_max_scale"])
        _, state["dash_max_scale"] = imgui.slider_float(
            "dash_max_scale",
            min(max(state["dash_max_scale"], 1.0), state["dash_scale_step"]),
            1.0,
            max(state["dash_scale_step"], 1.0001),
        )
        imgui.end_disabled()
        _, state["dash_scale_hysteresis"] = imgui.slider_float(
            "dash_scale_hysteresis", state["dash_scale_hysteresis"], 0.0, 0.5
        )
        imgui.set_item_tooltip(
            "How far past a step boundary the view must go before the dash\n"
            "size follows, in fractions of a step. Set it to 0, tick 'animate\n"
            "zoom' and watch a size sitting near a boundary flicker: that is\n"
            "what the default 0.1 exists to prevent. The cost is that the\n"
            "dashes may stray that much further from the size asked for."
        )
        imgui.end_disabled()
        if quantized and THICKNESS_SPACES[state["thickness_space"]] != "screen":
            imgui.text_wrapped(
                "Ignored: the pattern is anchored to the object already unless "
                'thickness_space is "screen".'
            )

        imgui.separator_text("Pattern (both halves)")
        _, state["pattern"] = imgui.combo(
            "dash_pattern",
            state["pattern"],
            [name for name, _ in DASH_PATTERNS],
            len(DASH_PATTERNS),
        )
        _, state["dash_offset"] = imgui.slider_float(
            "dash_offset", state["dash_offset"], 0.0, 4.0
        )
        _, state["thickness"] = imgui.slider_float(
            "thickness", state["thickness"], 1.0, 24.0
        )
        _, state["thickness_space"] = imgui.combo(
            "thickness_space",
            state["thickness_space"],
            THICKNESS_SPACES,
            len(THICKNESS_SPACES),
        )
        _, state["loop"] = imgui.checkbox("loop", state["loop"])
        _, state["aa"] = imgui.checkbox("aa", state["aa"])

        imgui.separator_text("Camera")
        _, state["camera_3d"] = imgui.checkbox("3D camera", state["camera_3d"])
        imgui.set_item_tooltip(
            "Switch between a perspective camera with orbit and an\n"
            "orthographic one with pan/zoom, in place. The position,\n"
            "orientation and framing are kept, so nothing changes size or\n"
            "moves off screen -- but a scene with depth does look different\n"
            "under a parallel projection, which is rather the point: the\n"
            "cubes' edges stop converging.\n\n"
            "The same toggle is in line_dashing_interactive.py, which starts\n"
            "on the other side of it."
        )
        imgui.begin_disabled(not state["camera_3d"])
        _, state["fov"] = imgui.slider_float("fov", state["fov"], 1.0, 120.0)
        imgui.set_item_tooltip(
            "The larger the fov, the more the near and far parts of a cube\n"
            "disagree about what one dash should measure. The camera is\n"
            "dollied to keep the framing, so this changes the perspective\n"
            "without changing how big anything is."
        )
        imgui.end_disabled()
        if imgui.button("reset view"):
            camera.set_state(home)
            set_fov(state["fov"] if state["camera_3d"] else 0.0)

        imgui.separator_text("Renderer")
        _, state["ppaa"] = imgui.combo(
            "ppaa", state["ppaa"], PPAA_MODES, len(PPAA_MODES)
        )
        _, state["pixel_filter"] = imgui.combo(
            "pixel_filter", state["pixel_filter"], PIXEL_FILTERS, len(PIXEL_FILTERS)
        )
        _, state["pixel_ratio"] = imgui.combo(
            "logical:physical",
            state["pixel_ratio"],
            [f"1:{r:g}" for r in PIXEL_RATIOS],
            len(PIXEL_RATIOS),
        )
        pw, ph = renderer.physical_size
        imgui.text(f"physical: {pw} x {ph}")

        imgui.pop_item_width()
    imgui.end()

    draw_overlay()


gui_renderer.set_gui(draw_imgui)


def animate():
    apply_state()
    width, height = canvas.get_logical_size()
    view_w = (width - PANEL_WIDTH) / 2
    # Keep the controller's idea of the panes in step with the window.
    view_left.rect = 0, 0, view_w, height
    view_right.rect = view_w, 0, view_w, height
    renderer.render(scene_left, camera, flush=False, rect=(0, 0, view_w, height))
    renderer.render(scene_right, camera, flush=False, rect=(view_w, 0, view_w, height))
    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    renderer.request_draw(animate)
    loop.run()
