"""
Test the cumulative distance that the line shader bakes to parametrize dashes.

The buffer is checked directly (rather than via a screenshot) because the
values have exact expected answers, which makes the assertions sharp.
"""

import numpy as np
import pytest
import pygfx as gfx
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


NAN = np.full((1, 3), np.nan, np.float32)


def regular_polygon(n, x=0.0, r=1.0):
    """The nodes of a regular n-gon, without repeating the first one."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


def polygon_side_length(n, r):
    return 2 * r * np.sin(np.pi / n)


def bake(positions, **material_kwargs):
    """Bake the line distance buffer for the given positions, and return it."""
    material_kwargs.setdefault("thickness_space", "model")
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(thickness=1, dash_pattern=[2, 2], **material_kwargs),
    )
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))
    return shader.line_distance_buffer.data


def test_cumdist_of_a_plain_line():
    positions = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist, [0, 3, 7])


def test_cumdist_restarts_at_each_line_piece():
    """Each nan-separated piece starts its dash pattern from scratch.

    Without this, the dash phase of a piece is offset by the total length of
    all the pieces before it, so the dashes of the later pieces race whenever
    something changes those lengths (such as the zoom level).
    """
    positions = np.vstack(
        [
            np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32),
            NAN,
            np.array([[0, 0, 0], [5, 0, 0]], np.float32),
            NAN,
            NAN,  # successive nans must not confuse the piece detection
            np.array([[0, 0, 0], [0, 2, 0]], np.float32),
        ]
    ).astype(np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist[0:3], [0, 3, 7])
    assert np.allclose(cumdist[4:6], [0, 5])
    assert np.allclose(cumdist[8:10], [0, 2])


def test_cumdist_closes_a_loop():
    """The node that closes a loop holds the cumdist of the *whole* loop.

    If it held the cumdist of the loop's first node instead, the shader would
    measure the closing segment as if it spanned the entire loop, and its
    dashes would come out that many times too dense. See gh-1103.
    """
    n, r = 5, 10.0
    side = polygon_side_length(n, r)
    cumdist = bake(regular_polygon(n, r=r), loop=True)

    # One extra element, to hold the cumdist of the closed loop
    assert len(cumdist) == n + 1
    assert np.allclose(cumdist, side * np.arange(n + 1))


def test_cumdist_closes_several_loops():
    """Every loop closes on itself, and each one starts from zero."""
    shapes = [(4, 10.0), (3, 5.0), (6, 7.0)]
    positions = np.vstack(
        [
            x
            for n, r in shapes
            for x in (regular_polygon(n, r=r), NAN)
            # the last nan is harmless: a trailing nan is not a loop
        ]
    ).astype(np.float32)
    cumdist = bake(positions, loop=True)

    i = 0
    for n, r in shapes:
        side = polygon_side_length(n, r)
        assert np.allclose(cumdist[i : i + n + 1], side * np.arange(n + 1))
        i += n + 1


def test_cumdist_honors_the_draw_range():
    """Loops are detected relative to the draw range, at the right nodes."""
    n, r = 4, 10.0
    side = polygon_side_length(n, r)
    positions = np.vstack([NAN, NAN, regular_polygon(n, r=r), NAN]).astype(np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1, dash_pattern=[2, 2], loop=True, thickness_space="model"
        ),
    )
    line.geometry.positions.draw_range = 2, n + 1
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))

    cumdist = shader.line_distance_buffer.data
    assert np.allclose(cumdist[2 : 2 + n + 1], side * np.arange(n + 1))


def test_cumdist_with_nonfinites_in_other_thickness_spaces():
    """Nans must not trip up the transform to world/screen space."""
    positions = np.vstack([regular_polygon(4, r=10.0), NAN]).astype(np.float32)
    for thickness_space in ["model", "world", "screen"]:
        cumdist = bake(positions, loop=True, thickness_space=thickness_space)
        assert np.all(np.isfinite(cumdist))
        assert cumdist[0] == 0
        assert cumdist[4] > cumdist[3] > 0


# ----- quantized dash scaling


def bake_quantized(positions, view_width, thickness=10.0, logical_size=(1000, 1000)):
    """Bake with dash_scaling='quantized' at a given ortho camera width."""
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=thickness,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    camera = gfx.OrthographicCamera(view_width, view_width)
    shader = LineShader(line)
    shader.bake_function(line, camera, logical_size)
    return shader.line_distance_buffer.data.copy(), shader._dash_levels[0]


def test_dash_scaling_only_applies_to_screen_space():
    """In model/world space the pattern is anchored to the object already."""
    positions = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    for thickness_space in ["model", "world"]:
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=1,
                dash_pattern=[2, 2],
                thickness_space=thickness_space,
                dash_scaling="quantized",
            ),
        )
        shader = LineShader(line)
        assert shader["dash_scaling"] == "continuous"
        assert shader["cumdist_space"] == thickness_space


def test_quantized_period_is_a_power_of_two_of_model_units():
    """One dash unit spans 2**level model units, whatever the zoom."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    for view_width, expected_level in [(400, 2), (200, 1), (100, 0), (50, -1)]:
        cumdist, level = bake_quantized(positions, view_width)
        assert level == expected_level
        # cumdist is in dash units, so the far node sits at length / 2**level
        assert np.allclose(cumdist[1], 100.0 / 2.0**level)


def test_quantized_dashes_split_rather_than_slide():
    """Every dash edge of a coarser level survives at the next finer level.

    This is the property that makes the dashes appear to split in two rather
    than travel along the line when the view scale changes.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    dash_size = 4.0

    def edges(view_width):
        cumdist, _ = bake_quantized(positions, view_width)
        # phase is linear in x along this straight line
        model_per_period = dash_size * 100.0 / cumdist[1]
        return np.arange(0, 100.0 + 1e-9, model_per_period)

    previous = None
    for view_width in np.linspace(320, 160, 17):  # a smooth 2x zoom in
        current = edges(view_width)
        if previous is not None:
            # every old edge must still be an edge, to within float error
            distance = np.abs(previous[:, None] - current[None, :]).min(axis=1)
            assert distance.max() < 1e-3, f"a dash edge moved by {distance.max()}"
        previous = current


def test_quantized_on_screen_size_stays_near_the_requested_one():
    """Rounding the log2 keeps the dash unit within sqrt(2) of `thickness`."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000
    for view_width in np.geomspace(40, 400, 25):
        _, level = bake_quantized(positions, view_width, thickness=thickness)
        model_units_per_pixel = view_width / logical
        on_screen = 2.0**level / model_units_per_pixel
        assert (
            thickness / np.sqrt(2) - 1e-6 <= on_screen <= thickness * np.sqrt(2) + 1e-6
        )


def test_quantized_level_does_not_flicker_at_a_boundary():
    """The level snap has hysteresis, so a view parked on a boundary is stable.

    Without it, jitter of a fraction of a percent flips the level on more than
    half of all frames, and the dashes stutter between splitting and merging.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)
    # thickness * view_width / logical_size == 2 ** (level + 0.5) at a boundary
    boundary = 100 * 2**0.5
    rng = np.random.default_rng(0)

    levels = []
    for _ in range(200):
        view_width = boundary * (1 + rng.uniform(-0.05, 0.05))
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        levels.append(shader._dash_levels[0])
    assert len(set(levels)) == 1, f"level flickered between {sorted(set(levels))}"


def test_dash_scale_hysteresis_is_a_material_property():
    """Turning it off must bring the flicker back, and up must damp harder.

    The default exists to stop a segment sitting near a step boundary from
    flipping between two sizes frame to frame. Setting it to zero makes the snap
    a plain threshold, which is the behaviour it was added to replace.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)

    def level_changes(hysteresis):
        material = gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
            dash_scale_hysteresis=hysteresis,
        )
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        shader = LineShader(line)

        # Park the view exactly on a boundary and jitter it.
        boundary = 100 * 2**0.5
        rng = np.random.default_rng(0)
        levels = []
        for _ in range(200):
            width = boundary * (1 + 0.01 * rng.uniform(-1, 1))
            camera = gfx.OrthographicCamera(width, width)
            camera.set_view_size(1000, 1000)
            shader.bake_function(line, camera, (1000, 1000))
            levels.append(float(shader._dash_levels[0]))
        return int(np.count_nonzero(np.diff(levels)))

    assert gfx.LineMaterial().dash_scale_hysteresis == 0.1
    assert level_changes(0.1) == 0, "the default must hold the size steady"
    assert level_changes(0.0) > 10, "without it, a plain threshold must flicker"


def test_quantized_level_still_follows_a_real_zoom():
    """Hysteresis must delay a level change, not prevent it."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)
    levels = []
    for view_width in np.geomspace(400, 50, 60):  # a 3-octave zoom in
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        levels.append(shader._dash_levels[0])
    # Monotonically decreasing, one step at a time, spanning three octaves
    steps = np.diff(levels)
    assert set(np.unique(steps)) <= {0, -1}
    assert levels[0] - levels[-1] == 3


def test_dash_max_scale_places_the_size_range():
    """`dash_max_scale` chooses where the factor-of-two band sits.

    The band is always exactly one octave wide -- that is what makes a level
    change split each dash rather than move it -- so the only freedom is where
    it sits relative to the size that `dash_pattern` asks for.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000
    overshoot = (
        2 ** gfx.LineMaterial().dash_scale_hysteresis
    )  # hysteresis lets it stray this much

    for dash_max_scale in [1.0, 2**0.5, 2.0]:
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=thickness,
                dash_pattern=[2, 2],
                thickness_space="screen",
                dash_scaling="quantized",
                dash_max_scale=dash_max_scale,
            ),
        )
        shader = LineShader(line)
        ratios, levels = [], []
        for view_width in np.geomspace(400, 25, 400):  # four octaves of zoom
            shader.bake_function(
                line, gfx.OrthographicCamera(view_width, view_width), (logical, logical)
            )
            ratios.append(
                2.0 ** shader._dash_levels[0] / (view_width / logical) / thickness
            )
            levels.append(shader._dash_levels[0])

        # the level only ever steps down, one at a time
        assert set(np.diff(levels)) <= {0, -1}
        # and the size stays inside the octave, give or take the hysteresis
        assert min(ratios) >= dash_max_scale / 2 / overshoot - 1e-6
        assert max(ratios) <= dash_max_scale * overshoot + 1e-6
        # the band really is a factor of two wide, not something narrower
        assert max(ratios) / min(ratios) > 1.5


def test_dash_max_scale_default_matches_round():
    """The default sqrt(2) is exactly the `round(log2(...))` behaviour."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    for view_width, expected_level in [(400, 2), (200, 1), (100, 0), (50, -1)]:
        _, level = bake_quantized(positions, view_width)
        assert level == expected_level


def test_dash_scaling_parameters_are_validated():
    material = gfx.LineMaterial(dash_pattern=[2, 2])
    assert material.dash_scale_step == 2.0
    assert material.dash_max_scale is None  # i.e. centred on the requested size
    with pytest.raises(ValueError):
        material.dash_scale_step = 0.9
    with pytest.raises(ValueError):
        material.dash_max_scale = 0.9


def test_dash_scale_step_one_reproduces_continuous():
    """The old behaviour is a value of the new parameter, not a separate mode.

    Exact at the nodes for any camera and any geometry, since the scale is
    taken per segment. See test_dash_scale_step_one_reproduces_continuous_in_3d
    for the case that used to be about 50% out.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness = 10.0

    def far_node(dash_scaling, view_width, step=1.0):
        material = gfx.LineMaterial(
            thickness=thickness,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling=dash_scaling,
        )
        if dash_scaling == "quantized":
            material.dash_scale_step = step
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        shader = LineShader(line)
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        value = shader.line_distance_buffer.data[1]
        # continuous keeps screen pixels and divides by thickness in the
        # shader; quantized bakes dash units directly
        return value / thickness if dash_scaling == "continuous" else value

    for view_width in np.geomspace(400, 40, 40):
        reference = far_node("continuous", view_width)
        assert np.isclose(far_node("quantized", view_width, 1.0), reference, rtol=1e-5)


def test_dash_scale_step_dials_smoothly_away_from_continuous():
    """A larger step departs further from the requested size, monotonically."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000

    def worst_deviation(step):
        material = gfx.LineMaterial(
            thickness=thickness,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        )
        material.dash_scale_step = step
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        shader = LineShader(line)
        worst = 0.0
        for view_width in np.geomspace(400, 40, 60):
            shader.bake_function(
                line, gfx.OrthographicCamera(view_width, view_width), (logical, logical)
            )
            ratio = shader._dash_units[0] / (thickness * view_width / logical)
            worst = max(worst, abs(ratio - 1))
        return worst

    deviations = [worst_deviation(step) for step in [1.0, 1.1, 1.5, 2.0]]
    assert deviations[0] < 1e-6  # step 1 is exactly the requested size
    assert deviations == sorted(deviations)  # and it grows with the step


def test_dash_scale_step_sets_the_width_of_the_size_range():
    """The dash size ranges over exactly one step, whatever the step is."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000
    overshoot = 2 ** gfx.LineMaterial().dash_scale_hysteresis

    for step in [1.5, 2.0, 3.0]:
        material = gfx.LineMaterial(
            thickness=thickness,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        )
        material.dash_scale_step = step
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        shader = LineShader(line)
        ratios = []
        for view_width in np.geomspace(400, 400 / step**4, 400):
            shader.bake_function(
                line, gfx.OrthographicCamera(view_width, view_width), (logical, logical)
            )
            ratios.append(shader._dash_units[0] / (thickness * view_width / logical))
        observed = max(ratios) / min(ratios)
        assert step / overshoot < observed <= step * overshoot


def test_quantized_levels_change_rarely_even_though_the_bake_runs():
    """The bake runs every frame; what has to be stable is the level.

    The scale is per segment, so it depends on where the camera is and the bake
    cannot be skipped the way it could when one factor served the whole object.
    What makes the dashes hold still is not the bake being skipped but the level
    being snapped: a small zoom must leave it alone, a large one must move it.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)

    def level(view_width):
        camera = gfx.OrthographicCamera(view_width, view_width)
        camera.set_view_size(1000, 1000)
        shader.bake_function(line, camera, (1000, 1000))
        return float(shader._dash_levels[0])

    first = level(200)
    assert level(190) == first, "a small zoom should not change the level"
    assert level(100) != first, "a 2x zoom should change the level"


def test_quantized_scale_is_per_segment_not_per_object():
    """A foreshortened segment must get its own dash size, not the object's.

    Two segments of equal *model* length, one across the view and one running
    away from the camera, cover very different numbers of pixels. A single
    factor for the whole object gives the receding one far too many dashes,
    which is what an edge-on cube used to look like.
    """
    # One piece per segment, so that each is measured on its own.
    positions = np.vstack(
        [
            np.array([[-80, 0, 0], [80, 0, 0]], np.float32),  # across the view
            NAN,
            np.array([[0, 0, 60], [0, 0, -100]], np.float32),  # running away
        ]
    ).astype(np.float32)

    camera = gfx.PerspectiveCamera(50, 1.0)
    camera.local.position = (0, 120, 260)
    camera.look_at((0, 0, 0))
    camera.set_view_size(600, 600)

    def bake_with(**kwargs):
        material = gfx.LineMaterial(
            thickness=10, dash_pattern=[2, 2], thickness_space="screen", **kwargs
        )
        line = gfx.Line(gfx.Geometry(positions=positions), material)
        shader = LineShader(line)
        shader.bake_function(line, camera, (600, 600))
        data = shader.line_distance_buffer.data
        # phase spanned by each of the two pieces
        return float(data[1]), float(data[4])

    across_c, away_c = bake_with()
    across_q, away_q = bake_with(dash_scaling="quantized", dash_scale_step=1.0)

    # Continuous bakes screen pixels; the shader divides by thickness. At step 1
    # the quantized path must land on exactly the same phase for both segments.
    assert across_q == pytest.approx(across_c / 10.0, rel=1e-4)
    assert away_q == pytest.approx(away_c / 10.0, rel=1e-4)

    # And the receding segment must hold far fewer periods than the other,
    # because it covers far fewer pixels despite being longer in model space.
    assert away_q < across_q


def test_dash_scale_step_one_reproduces_continuous_in_3d():
    """The equivalence must hold for an object with depth, from any angle.

    This is what the per-segment scale buys. With one factor per object it was
    about 50% out on a cube even under an orthographic camera, because a cube
    spans depth from every angle and one factor cannot serve its near and far
    edges at once.
    """
    corners = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
    pieces = [
        np.array([[x, y, -50] for x, y in corners], np.float32),
        np.array([[x, y, 50] for x, y in corners], np.float32),
        *[np.array([[x, y, -50], [x, y, 50]], np.float32) for x, y in corners],
    ]
    stacked = [pieces[0]]
    for piece in pieces[1:]:
        stacked += [NAN, piece]
    cube = np.vstack(stacked).astype(np.float32)

    def phase(camera, **kwargs):
        material = gfx.LineMaterial(
            thickness=8,
            dash_pattern=[2, 2],
            thickness_space="screen",
            loop=True,
            **kwargs,
        )
        line = gfx.Line(gfx.Geometry(positions=cube), material)
        shader = LineShader(line)
        shader.bake_function(line, camera, (600, 600))
        data = shader.line_distance_buffer.data.copy()
        return data if kwargs.get("dash_scaling") == "quantized" else data / 8.0

    for position in [(120, 90, 200), (260, 0, 20), (30, 240, 60), (0, 0, 300)]:
        camera = gfx.PerspectiveCamera(50, 1.0)
        camera.local.position = position
        camera.look_at((0, 0, 0))
        camera.set_view_size(600, 600)

        reference = phase(camera)
        quantized = phase(camera, dash_scaling="quantized", dash_scale_step=1.0)
        usable = np.isfinite(reference) & np.isfinite(quantized) & (reference > 1e-6)
        deviation = np.max(
            np.abs(quantized[usable] - reference[usable]) / reference[usable]
        )
        assert deviation < 1e-4, f"from {position}: {deviation:.4%}"


# The period of the [2, 2] pattern that `bake` uses, and its final gap.
PERIOD = 4.0
LAST_GAP = 2.0


def test_dash_fit_stretches_a_loop_to_whole_periods():
    """A fitted loop holds an exact whole number of periods, so it has no seam.

    Unfitted, the pattern arrives back at its starting point part-way through a
    period, and the stroke that closes the loop is the wrong length.
    """
    positions = regular_polygon(99, r=3.3)
    perimeter = 99 * polygon_side_length(99, 3.3)

    exact = bake(positions, loop=True)
    assert np.allclose(exact[-1], perimeter)
    assert not np.isclose(exact[-1] / PERIOD, round(exact[-1] / PERIOD), atol=1e-3), (
        "this geometry is meant to be an awkward fit, so that the test can bite"
    )

    fitted = bake(positions, loop=True, dash_fit="stretch")
    assert np.isclose(fitted[-1] / PERIOD, round(fitted[-1] / PERIOD))
    # The stretch is only the small factor needed to reach the nearest fit.
    assert 0.9 < fitted[-1] / perimeter < 1.1


def test_dash_fit_ends_an_open_piece_on_a_stroke():
    """An open piece is fitted to a whole number of periods less the final gap.

    Fitting it to whole periods instead would leave it ending in empty space,
    since the piece starts at the beginning of a stroke.
    """
    positions = np.array([[0, 0, 0], [10.3, 0, 0]], np.float32)
    fitted = bake(positions, dash_fit="stretch")
    assert np.allclose(fitted[-1], 10.0)  # 3 periods less the final gap
    assert np.isclose((fitted[-1] + LAST_GAP) / PERIOD, 3.0)


def test_dash_fit_is_per_piece():
    """Each nan-separated piece is fitted on its own.

    This is the whole point of fitting: pieces have different lengths, so a
    single global factor could not make all of them come out whole.
    """
    positions = np.vstack(
        [
            np.array([[0, 0, 0], [10.3, 0, 0]], np.float32),  # open
            NAN,
            regular_polygon(3, r=1.0),  # closed, ~5.196 long
            NAN,
            regular_polygon(4, r=1.7),  # closed, ~9.617 long
            NAN,
            regular_polygon(99, r=3.3),  # closed, ~20.73 long
        ]
    ).astype(np.float32)
    fitted = bake(positions, loop=True, dash_fit="stretch")

    # The span of a piece sits at its last node: the connector for a loop, and
    # the final node for the open piece.
    assert np.allclose(fitted[1], 3 * PERIOD - LAST_GAP)  # open
    assert np.allclose(fitted[6], 1 * PERIOD)  # triangle
    assert np.allclose(fitted[11], 2 * PERIOD)  # square
    assert np.allclose(fitted[-1], 5 * PERIOD)  # 99-gon


def test_dash_fit_stretches_a_short_piece_to_one_period():
    """A piece shorter than one period still gets a whole period, not zero."""
    positions = regular_polygon(3, r=0.05)  # perimeter ~0.26, well under a period
    fitted = bake(positions, loop=True, dash_fit="stretch")
    assert np.allclose(fitted[-1], PERIOD)


def test_dash_fit_exact_leaves_the_cumdist_alone():
    """The default must be the plain arc length, i.e. bit-identical to no fitting."""
    positions = np.vstack(
        [
            np.array([[0, 0, 0], [10.3, 0, 0]], np.float32),
            NAN,
            regular_polygon(4, r=1.7),
        ]
    ).astype(np.float32)
    assert np.array_equal(
        bake(positions, loop=True), bake(positions, loop=True, dash_fit="exact")
    )
    assert np.allclose(bake(positions, loop=True)[1], 10.3)


def test_dash_fit_handles_a_pattern_that_is_all_gap():
    """A pattern with no stroke has no stroke end to finish on.

    `dash_pattern=[0, 4]` is a legal (if odd) pattern. The open-piece target of
    `n * period - last_gap` is zero for it, which would scale the whole piece to
    nothing; whole periods are used instead.
    """
    positions = np.array([[0, 0, 0], [10.3, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1,
            dash_pattern=[0, 4],
            thickness_space="model",
            dash_fit="stretch",
        ),
    )
    shader = LineShader(line)
    shader.bake_function(line, gfx.OrthographicCamera(100, 100), (100, 100))
    fitted = shader.line_distance_buffer.data

    assert fitted[-1] > 0
    assert np.isclose(fitted[-1] / 4.0, round(fitted[-1] / 4.0))


def test_dash_fit_counts_ride_the_step_ladder_under_quantization():
    """Under quantization the period count is snapped to a power of the step.

    The nearest whole number would break the property that quantization exists
    for: a level change must split each dash rather than move it, and a count of
    7 cannot halve to a count of 3.5. A count on the power ladder always can, so
    the dash edges of one level stay a strict subset of the next level's.
    """
    positions = regular_polygon(99, r=3.3)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=8,
            dash_pattern=[2, 2],
            thickness_space="screen",
            loop=True,
            dash_scaling="quantized",
            dash_fit="stretch",
        ),
    )
    shader = LineShader(line)

    counts = []
    for exp in np.linspace(0, 4, 33):  # a smooth 16x zoom
        width = 100 / 2**exp
        shader.bake_function(line, gfx.OrthographicCamera(width, width), (1000, 1000))
        counts.append(shader.line_distance_buffer.data[-1] / PERIOD)
    counts = np.array(counts)

    assert np.allclose(counts, np.round(counts)), "every count must be whole"
    assert np.allclose(np.log2(counts), np.round(np.log2(counts))), (
        "and must sit on the power-of-two ladder"
    )
    # Frame to frame the count either holds or doubles, never lands elsewhere.
    ratios = counts[1:] / counts[:-1]
    assert np.all(np.isclose(ratios, 1.0) | np.isclose(ratios, 2.0))
    assert counts.max() > counts.min(), "the sweep must actually cross a level"

    # Dash edges sit at multiples of 1/count around the loop, so a coarser
    # level's edges are exactly a subset of a finer level's.
    coarse, fine = int(counts.min()), int(counts.max())
    assert set(np.arange(coarse) / coarse) <= set(np.arange(fine) / fine)


def test_dash_fit_rebakes_when_the_pattern_or_thickness_changes():
    """The fit depends on inputs the unfitted model-space cumdist does not.

    In model space the bake short-circuits on the positions alone. The fit also
    depends on the pattern and the thickness, so those have to join the hash or
    a change to either is silently ignored.
    """
    positions = regular_polygon(99, r=3.3)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1,
            dash_pattern=[2, 2],
            thickness_space="model",
            loop=True,
            dash_fit="stretch",
        ),
    )
    shader = LineShader(line)
    camera = gfx.OrthographicCamera(100, 100)

    def bake_span():
        shader.bake_function(line, camera, (1000, 1000))
        return shader.line_distance_buffer.data[-1]

    first = bake_span()
    line.material.thickness = 2
    assert bake_span() != first, "a thickness change must re-fit"
    line.material.thickness = 1
    assert bake_span() == first
    line.material.dash_pattern = [6, 2]
    assert bake_span() != first, "a pattern change must re-fit"


def spread(positions, **material_kwargs):
    """Bake with dash_fit='spread' and return (cumdist, per-node gap scale)."""
    material_kwargs.setdefault("thickness_space", "model")
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1, dash_pattern=[2, 2], dash_fit="spread", **material_kwargs
        ),
    )
    shader = LineShader(line)
    shader.bake_function(line, gfx.OrthographicCamera(100, 100), (100, 100))
    return (
        shader.line_distance_buffer.data.copy(),
        shader.line_gap_scale_buffer.data.copy(),
    )


def test_dash_fit_spread_leaves_the_strokes_alone():
    """Only the gaps give, so the cumdist must be the plain arc length.

    `stretch` reaches its fit by scaling the phase, which changes the strokes
    with everything else. `spread` must not touch it: the strokes keep exactly
    the size the pattern asks for, and the whole adjustment is carried by the
    gap scale instead.
    """
    positions = np.array([[0, 0, 0], [25.3, 0, 0]], np.float32)
    cumdist, gap_scale = spread(positions)

    assert np.allclose(cumdist, bake(positions)), "the phase must be untouched"
    assert np.allclose(gap_scale, gap_scale[0]), "one scale for the whole piece"
    assert gap_scale[0] > 1.0, "this piece needs its gaps widened to fit"


@pytest.mark.parametrize("length", [9.0, 12.5, 16.0, 19.9, 20.0, 25.3, 41.7, 100.0])
def test_dash_fit_spread_never_asks_for_a_smaller_pattern(length):
    """The pattern is a floor: the gaps may only ever grow.

    This is what separates it from `stretch`, which reaches the *nearest* whole
    number of periods and so compresses a piece about half the time.
    """
    positions = np.array([[0, 0, 0], [length, 0, 0]], np.float32)
    _, gap_scale = spread(positions)
    assert gap_scale.min() >= 1.0, f"gaps were narrowed to {gap_scale.min()}"


def test_dash_fit_spread_fits_a_whole_number_of_periods():
    """With the widened gaps, the piece must hold a whole number of periods.

    The strokes total 2 and the gaps 2 per period, so a piece of phase span S
    holds n periods of `2 + 2 * gap_scale` less the final widened gap.
    """
    for length in (12.5, 19.9, 25.3, 41.7, 100.0):
        positions = np.array([[0, 0, 0], [length, 0, 0]], np.float32)
        cumdist, gap_scale = spread(positions)
        span, k = float(cumdist[-1]), float(gap_scale[0])
        period = 2 + 2 * k
        # Open piece: it ends on a stroke end, so the last gap is not included.
        count = (span + 2 * k) / period
        assert count == pytest.approx(round(count)), (
            f"length {length}: {count} periods, not a whole number"
        )
        assert round(count) >= 1


def test_dash_fit_spread_adds_a_stroke_instead_of_squeezing():
    """As a piece grows the gaps widen, then a stroke appears and they snap back.

    The count comes from a floor rather than a round, which is exactly what
    makes the pattern a floor too.
    """
    counts, scales = [], []
    for length in np.arange(20.0, 44.0, 1.0):
        positions = np.array([[0, 0, 0], [length, 0, 0]], np.float32)
        cumdist, gap_scale = spread(positions)
        k = float(gap_scale[0])
        scales.append(k)
        counts.append(round((float(cumdist[-1]) + 2 * k) / (2 + 2 * k)))

    assert max(counts) > min(counts), "the sweep must cross a stroke being added"
    for i in range(1, len(counts)):
        if counts[i] > counts[i - 1]:
            assert scales[i] < scales[i - 1], "adding a stroke must narrow the gaps"
        else:
            assert scales[i] >= scales[i - 1] - 1e-6, "otherwise they only widen"


def test_dash_fit_is_validated():
    material = gfx.LineMaterial()
    assert material.dash_fit == "exact"
    material.dash_fit = "stretch"
    assert material.dash_fit == "stretch"
    with pytest.raises(ValueError):
        material.dash_fit = "squash"


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            func()
