"""
Dashing, interactively
======================

Live controls for the two dashing questions that only show up once something
moves or once a line closes: what the dashes do when you zoom (``dash_scaling``,
``dash_scale_step``, ``dash_max_scale``) and whether the pattern is made to fit
the line (``dash_fit``).

Split view: **left is pygfx's current behaviour, right is whatever the panel
says**. One camera drives both halves, so the same shapes sit at the same place
on screen in both and the only difference is the material. Everything except the
fit and the quantization is mirrored onto both sides, so you only ever see the
one difference you are looking at.

The five shapes -- a straight line, a triangle, a square, a pentagon, and a
99-gon standing in for a circle -- give five piece lengths at once, which is what
makes a per-piece behaviour legible.

**dash_fit.** By default the pattern keeps exactly the size you asked for and
simply stops wherever the piece ends. Almost no piece is a whole number of
periods long, so a closed piece has a seam where the pattern comes back round to
its start: look at the top of each polygon on the left. Setting
``dash_fit="stretch"`` scales each piece by the small factor that makes a whole
number of periods span it, and the seam goes. It is per piece, so the five shapes
end up with five slightly different dash sizes -- that is the trade being made.
Open pieces are fitted to whole periods *less the final gap*, so the straight
line begins and ends with a full stroke rather than trailing off into space.

``dash_fit="spread"`` fits without ever asking for a smaller pattern. Reaching
the *nearest* whole number of periods means compressing a piece about half the
time, and ``"stretch"`` does; ``"spread"`` drops to the next count down and
widens only the gaps, leaving the strokes at exactly the size asked for. The
pattern is then a floor. Watch one shape while you zoom: the gaps open up, and
when there is room for another stroke it appears and they snap back.

**dash_scaling.** With ``thickness_space="screen"`` the pattern is measured in
screen pixels, so the number of dashes along a piece has to change as you zoom,
and each dash drifts in proportion to its distance from the start of its piece.
Tick "animate zoom" and watch the left half crawl. With ``"quantized"`` the
pattern is anchored to the object and only its period follows the view, in steps
of ``dash_scale_step``, so the dashes hold still and split.

The two combine: under quantization the fitted period count is snapped to a power
of the step rather than to the nearest whole number, so a level change still
splits each dash exactly rather than moving it.

**Getting back to today's behaviour.** Press **continuous**. It takes two
settings, not one: ``dash_fit=exact`` *and* ``dash_scale_step=1``, since a
stretched pattern differs from main whatever the scaling is set to. A step of 1
is the point of the family where quantization does nothing -- it is 1 and not 0
because the parameter is the ratio between successive dash sizes, not an
increment. Start there, then raise the step to dial quantization in.

**single object vs distinct objects.** The five shapes can be drawn either as one
``Line`` with nans between the pieces, or as five separate ``Line`` objects
sharing one material. That these look the same is the property the per-piece dash
phase exists to give, so the toggle is a check rather than a demonstration: flip
it and nothing should move.

They are not quite bit-identical, and it is worth knowing why. A later piece's
cumulative distance is summed straight through the lengths of the pieces before
it, and only then has its own starting offset subtracted, so it carries a couple
of float32 ulps that an object starting from zero does not. Measured, the two
agree to about 2e-7 relative, which is a handful of antialiased pixels at the
loop seams and nothing anywhere else.

**Renderer settings.** The panel also exposes what sits between the line shader
and the pixels: the post-processing antialiasing (``ppaa``), the reconstruction
filter of the downsample (``pixel_filter``), the line's own analytic ``aa``, and
whether the render is 1:1 or 1:2 against the window. Turn ``ppaa`` off and set
the ratio to 1:1 and what is left is what the shader actually emitted.

Drag to pan and scroll to zoom in **either** pane; one camera drives both, so
the halves always show the same view. See ``line_dashing_3d.py`` for the same
controls under a perspective camera.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import math
import time

import numpy as np
import pylinalg as la
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


TITLE = "Dashing, interactively"
CANVAS_SIZE = 1500, 820
PANEL_WIDTH = 360
LABEL_WIDTH = -170

# Half the canvas, less the panel. The camera is sized to match, so that at zoom
# 1 one model unit is one logical pixel in each pane.
VIEW_SIZE = (CANVAS_SIZE[0] - PANEL_WIDTH) / 2, CANVAS_SIZE[1]

# Five shapes in a 3 + 2 grid, which fits a half-width pane where a single row
# would not. The straight line takes the first slot.
RADIUS = 62.0
SLOTS = [(-165.0, 165.0), (0.0, 165.0), (165.0, 165.0), (-85.0, -125.0), (85.0, -125.0)]
POLYGONS = [3, 4, 5, 99]

DASH_PATTERNS = [
    ("2, 2", [2, 2]),
    ("4, 2", [4, 2]),
    ("1, 3", [1, 3]),
    ("6, 2, 2, 2", [6, 2, 2, 2]),
]
THICKNESS_SPACES = ["screen", "world", "model"]
DASH_SCALINGS = ["continuous", "quantized"]
DASH_FITS = ["exact", "stretch", "spread"]
GEOMETRY_MODES = ["single object (nans)", "distinct objects"]
PPAA_MODES = ["none", "fxaa", "ddaa"]
PIXEL_FILTERS = ["nearest", "linear", "tent", "disk", "bspline", "mitchell", "catmull"]
PIXEL_RATIOS = [1.0, 2.0]

AUTO_ZOOM_RANGE = 0.0, 3.0
AUTO_ZOOM_PERIOD = 16.0  # seconds for a full there-and-back


def polygon(n, cx, cy, r=RADIUS):
    # An even-sided polygon drawn from a vertex at the top reads as standing on
    # a corner (a square becomes a diamond), so turn it half a side; an odd one
    # already stands on a side, so leave it pointing up.
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + (np.pi / n if n % 2 == 0 else 0)
    return np.stack(
        [cx + r * np.sin(t), cy + r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


def pieces():
    """The five shapes, each as its own array of nodes."""
    (lx, ly), *rest = SLOTS
    straight = np.array([[lx - RADIUS, ly, 0], [lx + RADIUS, ly, 0]], np.float32)
    return [straight] + [
        polygon(n, x, y) for n, (x, y) in zip(POLYGONS, rest, strict=True)
    ]


def joined(parts):
    """The same pieces as one array, nan-separated."""
    nanpoint = np.full((1, 3), np.nan, np.float32)
    stacked = [parts[0]]
    for part in parts[1:]:
        stacked += [nanpoint, part]
    return np.vstack(stacked).astype(np.float32)


canvas = RenderCanvas(size=CANVAS_SIZE, title=TITLE)
renderer = gfx.WgpuRenderer(canvas)


def build_side(color):
    """One pane, with the shapes built both ways on top of each other."""
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
    parts = pieces()
    single = gfx.Line(gfx.Geometry(positions=joined(parts)), material)
    distinct = [gfx.Line(gfx.Geometry(positions=part), material) for part in parts]
    scene.add(single, *distinct)
    return scene, material, single, distinct


scene_left, control, control_single, control_distinct = build_side("#7af")
scene_right, live, live_single, live_distinct = build_side("#fd5")

# An orthographic camera in pygfx *is* a perspective one with fov 0 -- they are
# the same class -- so the 2D and 3D views here are one camera with two values
# of `fov`, and switching between them keeps the position, orientation and view
# extent untouched. What changes is the projection and which controller drives
# it. `width` is the extent at the reference depth, which the projection keeps
# fixed across a change of fov, so nothing jumps: measured, a box at that depth
# covers the same 10000 pixels at fov 0, 10, 35 and 70.
camera = gfx.PerspectiveCamera(0)
camera.width, camera.height = VIEW_SIZE

# One viewport per pane, and the controller is handed whichever one the pointer
# is over. It measures pan and zoom against the viewport's rect, so that rect has
# to be the pane actually under the cursor: registered on the renderer it would
# scale by the whole canvas width, and registered on one pane the other half
# would be dead, because `pointer_down` and `wheel` are gated on `is_inside`.
#
# That gating is also what keeps the panel usable: events over it fall in neither
# rect, so dragging a slider does not move the camera as well.
view_left = gfx.Viewport(renderer, rect=(0, 0, *VIEW_SIZE))
view_right = gfx.Viewport(renderer, rect=(VIEW_SIZE[0], 0, *VIEW_SIZE))
panzoom = gfx.PanZoomController(camera)
orbit = gfx.OrbitController(camera)


def active_controller():
    return orbit if state["camera_3d"] else panzoom


def standoff(fov):
    """How far back the camera must sit for `fov` to span the current height.

    Zero for an orthographic camera, which has no such distance: its rays are
    parallel, so it can sit in the plane it is looking at -- which is exactly
    where this one starts, and why switching to a perspective projection
    without moving it would put the scene at w = 0.
    """
    if fov <= 0:
        return 0.0
    # pygfx builds both projections from the same reference size, the mean of
    # the camera's width and height, and applies the same aspect correction to
    # each -- so that cancels, and equating the two extents at distance D gives
    # this. The zoom factor cancels as well, so a zoomed view switches cleanly.
    return 0.25 * (camera.width + camera.height) / math.tan(math.radians(fov) / 2)


def set_fov(fov):
    """Change the projection while keeping the framing.

    The camera is dollied along its own axis by the change in standoff, so the
    plane it was looking at still spans the same height afterwards. Nothing on
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
    "geometry": 0,
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
    # Seeded from the renderer's own defaults, so that merely opening the panel
    # cannot change how anything is drawn.
    "ppaa": PPAA_MODES.index(renderer.ppaa),
    "pixel_filter": PIXEL_FILTERS.index(renderer.pixel_filter),
    "pixel_ratio": min(
        range(len(PIXEL_RATIOS)),
        key=lambda i: abs(PIXEL_RATIOS[i] - renderer.pixel_ratio),
    ),
    "animate_zoom": False,
    # The camera is orthographic until this is ticked; see `apply_state`.
    "camera_3d": False,
    "fov": 45.0,
}
# What "reset" restores, captured before anything can change it.
OPENING_STATE = dict(state)
# What "continuous" restores: the point of the parameter family where
# quantization does nothing, which is pygfx's behaviour on main. Note it takes
# two of the three -- dash_fit has to be "exact" as well, since a stretched
# pattern differs from main whatever the scaling is set to.
CONTINUOUS_STATE = OPENING_STATE | {
    "dash_fit": DASH_FITS.index("exact"),
    "dash_scaling": DASH_SCALINGS.index("quantized"),
    "dash_scale_step": 1.0,
}
applied = {}


def get_zoom_exp():
    """The current view scale, as a power of two.

    Read back from the camera rather than stored, so that the GUI and the mouse
    have one shared handle on the view instead of fighting over two.
    """
    return float(np.log2(VIEW_SIZE[0] / camera.width))


def set_zoom_exp(exp):
    factor = 2.0**exp
    camera.width = VIEW_SIZE[0] / factor
    camera.height = VIEW_SIZE[1] / factor


def apply_state():
    """Push the panel onto the right pane, and the shared bits onto both.

    The guard matters: the uniform-backed setters re-upload whether or not the
    value differs, and this runs every frame.
    """
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

    single = GEOMETRY_MODES[state["geometry"]].startswith("single")
    for one, many in ((live_single, live_distinct), (control_single, control_distinct)):
        one.visible = single
        for line in many:
            line.visible = not single


def draw_overlay():
    """The two captions and the divider, drawn over the panes."""
    display = gui_renderer.backend.io.display_size
    view_w = (display.x - PANEL_WIDTH) / 2
    draw = imgui.get_foreground_draw_list()
    draw.add_line(
        imgui.ImVec2(view_w, 0),
        imgui.ImVec2(view_w, display.y),
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
            "Put every control back to how the example opened:\n"
            "dash_fit=spread, dash_scaling=quantized, dash_scale_step=2.\n"
            "Does not move the camera; use 'reset view' for that."
        )
        imgui.same_line()
        if imgui.button("continuous"):
            state.update(CONTINUOUS_STATE)
        imgui.set_item_tooltip(
            "Make the right pane match the left exactly, i.e. what pygfx does\n"
            "on main: dash_fit=exact, dash_scaling=quantized,\n"
            "dash_scale_step=1.\n\n"
            "A step of 1 is the point where quantization does nothing -- the\n"
            "size follows the view exactly, which is what continuous scaling\n"
            "means. It is 1 and not 0 because the parameter is the ratio\n"
            "between successive dash sizes, not an increment.\n\n"
            "Start here and raise dash_scale_step to dial quantization in."
        )

        imgui.separator_text("Fitting (right pane only)")
        _, state["dash_fit"] = imgui.combo(
            "dash_fit", state["dash_fit"], DASH_FITS, len(DASH_FITS)
        )

        imgui.separator_text("Quantization (right pane only)")
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
        # The property clamps to [1, step]; clamp the slider to match, so that
        # it shows the value that is actually in effect.
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

        imgui.separator_text("Pattern (both panes)")
        _, state["pattern"] = imgui.combo(
            "dash_pattern",
            state["pattern"],
            [name for name, _ in DASH_PATTERNS],
            len(DASH_PATTERNS),
        )
        _, state["dash_offset"] = imgui.slider_float(
            "dash_offset", state["dash_offset"], 0.0, 4.0
        )
        if state["dash_offset"] % 1.0:
            if quantized:
                imgui.text_wrapped(
                    "A fractional offset is a fraction of a period here, so it "
                    "slides the pattern at every level change."
                )
            if DASH_FITS[state["dash_fit"]] == "stretch":
                imgui.text_wrapped(
                    "A fractional offset moves the open piece off its stroke "
                    "ends; closed pieces still join up."
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
        _, state["aa"] = imgui.checkbox("aa (the line's own analytic AA)", state["aa"])

        imgui.separator_text("Geometry (both panes)")
        _, state["geometry"] = imgui.combo(
            "drawn as", state["geometry"], GEOMETRY_MODES, len(GEOMETRY_MODES)
        )
        imgui.text_wrapped(
            "Nothing should move; they agree to ~2e-7, a few pixels at the seams."
        )

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

        imgui.separator_text("View")

        _, state["camera_3d"] = imgui.checkbox("3D camera", state["camera_3d"])
        imgui.set_item_tooltip(
            "Switch between an orthographic camera with pan/zoom and a\n"
            "perspective one with orbit, in place: the position, orientation\n"
            "and view extent are kept, so the picture does not jump. On a flat\n"
            "scene like this one nothing visibly moves at the moment you tick\n"
            "it -- what changes is that dragging now rotates instead of\n"
            "panning, and that the dash scale starts to vary with depth once\n"
            "you tilt the view."
        )
        imgui.begin_disabled(not state["camera_3d"])
        _, state["fov"] = imgui.slider_float("fov", state["fov"], 1.0, 120.0)
        imgui.end_disabled()

        _, state["animate_zoom"] = imgui.checkbox("animate zoom", state["animate_zoom"])
        imgui.begin_disabled(state["animate_zoom"])
        changed, exp = imgui.slider_float("zoom (2**x)", get_zoom_exp(), -1.0, 5.0)
        if changed:
            set_zoom_exp(exp)
        imgui.end_disabled()
        if imgui.button("reset view"):
            camera.set_state(home)
            # `home` was captured while orthographic, so re-apply the standoff
            # for whatever the toggle currently says rather than leaving fov 0.
            set_fov(state["fov"] if state["camera_3d"] else 0.0)
        imgui.text(f"view scale: {2 ** get_zoom_exp():.2f} px per model unit")
        _, pattern = DASH_PATTERNS[state["pattern"]]
        imgui.text(f"dash size asked for: {state['thickness'] * pattern[0]:.1f} px")

        imgui.pop_item_width()
    imgui.end()

    draw_overlay()


gui_renderer.set_gui(draw_imgui)

start_time = time.perf_counter()


def animate():
    if state["animate_zoom"]:
        low, high = AUTO_ZOOM_RANGE
        phase = (time.perf_counter() - start_time) / AUTO_ZOOM_PERIOD
        # A cosine rather than a sawtooth, so the sweep slows to a stop at each
        # end instead of snapping back and hiding what just happened.
        set_zoom_exp(low + (high - low) * 0.5 * (1 - np.cos(2 * np.pi * phase)))

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
