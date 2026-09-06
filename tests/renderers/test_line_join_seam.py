"""
Characterise the seam a line draws across a broken join.

A line overlaps itself wherever a join is "broken" -- a corner too sharp to be
mitred, which the shader covers with the two segments' own caps instead. The
two faces are coplanar and disagree about coverage in the overlap, and the
depth test arbitrates between them. Under the default ``depth_compare="<"``
only the first fragment survives, so where it sits on an antialiased edge that
its sibling covers solidly, the pixel keeps a partial alpha: a dark hairline is
drawn across the inside of the corner.

These tests describe the defect rather than fix it. The requirement below is
marked xfail, strictly, so that whoever does fix it is told by a failing
XPASS rather than having to notice this file.

The measurement uses a *translucent* line, which is much the better probe:

* it is exact. Every inked pixel of a uniformly translucent line over a flat
  background must be one value, so any departure is a defect and its direction
  says which one -- darker means a fragment was dropped, brighter means one was
  composited twice.
* it is stable. The opaque seam depends on the pixel ratio (invisible at 1,
  224/255 at 2, 241/255 at 4), so an opaque test can silently pass. The
  translucent one shows the defect at every pixel ratio.

Note that dashing is what makes a 60 degree corner a broken join at all: the
shader mitres up to ``max_vec_mag``, which is 100 when solid but drops to 1.5
(about 90 degrees) when dashing. A *solid* 60 degree corner is properly mitred
and is clean.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


SIZE = 200
THICKNESS = 14.0
ALPHA = 0.4
# The criterion is uniformity, not a particular value: a line of one colour and
# one alpha over a flat background must composite to one value everywhere, and
# what that value is does not matter. (It is not 255*ALPHA, incidentally --
# compositing happens in linear space, so 0.4 comes out at 170, not 102.)
# Antialiasing makes the *edge* pixels legitimately different, so the interior
# is eroded before measuring, and a few levels of slack are allowed.
TOLERANCE = 4
# The dash phase at the corner is leg_length / THICKNESS in dash units, so this
# puts the corner one unit into a two-unit stroke: a dash sits astride it.
LEG_LENGTH = 70.0


def erode(mask, iterations=2):
    """Shrink a boolean mask by `iterations` pixels, in all 8 directions."""
    for _ in range(iterations):
        m = mask
        mask = np.pad(
            m[1:-1, 1:-1]
            & m[:-2, 1:-1]
            & m[2:, 1:-1]
            & m[1:-1, :-2]
            & m[1:-1, 2:]
            & m[:-2, :-2]
            & m[:-2, 2:]
            & m[2:, :-2]
            & m[2:, 2:],
            1,
        )
    return mask


def render_corner(angle_deg, *, dashed=True, alpha=ALPHA, pixel_ratio=None, **kwargs):
    """A translucent white corner on black, rendered offscreen.

    The corner sits at the origin with both legs running upwards, so that the
    inside of it -- the interesting part -- is in view.
    """
    half = np.radians(angle_deg / 2)
    dx, dy = LEG_LENGTH * np.sin(half), LEG_LENGTH * np.cos(half)
    positions = np.array([[-dx, dy, 0], [0, 0, 0], [dx, dy, 0]], np.float32)

    target = gfx.Texture(
        dim=2, size=(SIZE, SIZE, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    # The seam is the line shader's own. Post-processing AA would only blur the
    # evidence and make the numbers adapter-dependent.
    renderer.ppaa = "none"
    if pixel_ratio is not None:
        renderer.pixel_ratio = pixel_ratio

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=THICKNESS,
                color=(1, 1, 1, alpha),
                aa=True,
                dash_pattern=[2, 2] if dashed else (),
                **kwargs,
            ),
        )
    )

    camera = gfx.OrthographicCamera(SIZE, SIZE)
    camera.local.position = (0, 45, 0)
    renderer.render(scene, camera)
    return renderer.snapshot()[..., 0].astype(int)


def interior_range(image):
    """The (min, max) of the pixels strictly inside the ink."""
    interior = erode(image > 40)
    assert interior.any(), "nothing was drawn, so the measurement proves nothing"
    return int(image[interior].min()), int(image[interior].max())


@pytest.mark.xfail(strict=True, reason="the broken-join seam is not fixed yet")
@pytest.mark.parametrize("angle", [70, 60, 45, 30])
def test_no_seam_across_a_broken_join(angle):
    """A uniformly translucent line must render to one uniform value.

    This is the requirement. It currently fails with the interior spanning
    roughly 70..170 instead of a single value: a one-pixel dark line drawn
    across the inside of the corner.
    """
    low, high = interior_range(render_corner(angle))
    assert high - low <= TOLERANCE, (
        f"the inside of a {angle} degree corner is not uniform: {low}..{high}"
    )


@pytest.mark.parametrize("angle", [90, 70, 60, 45])
def test_a_mitred_join_is_clean(angle):
    """A solid line mitres these corners, and mitred joins do not seam.

    This is the control. It shows the defect belongs to the broken join and not
    to sharp corners as such, and it guards the measurement: if this ever fails,
    the harness is wrong rather than the shader.
    Thirty degrees and sharper is excluded: a solid corner that sharp has a
    small defect of its own (a spread of about 18), which `depth_compare` does
    not affect either way and which is not understood.
    """
    low, high = interior_range(render_corner(angle, dashed=False))
    assert high - low <= TOLERANCE, f"{angle} degrees: {low}..{high}"


def test_depth_compare_le_is_not_the_fix():
    """`depth_compare="<="` moves the error rather than removing it.

    It is the obvious one-line candidate, and it does remove the *dark* seam,
    because both coplanar fragments now survive instead of one being dropped.
    But surviving means both composite, so the overlap is painted twice and the
    seam comes back as a bright patch. On an opaque line that is invisible,
    which is exactly why the opaque metric must not be trusted here.

    Kept as a passing test so the candidate is not re-proposed.
    """
    plain_low, plain_high = interior_range(render_corner(60))
    low, high = interior_range(render_corner(60, depth_compare="<="))
    assert low > plain_low + 3 * TOLERANCE, "the dark seam should be gone"
    assert high > plain_high + 3 * TOLERANCE, (
        f"expected a bright patch from double compositing, got {low}..{high}"
    )


@pytest.mark.parametrize("pixel_ratio", [1, 2, 4])
def test_the_defect_is_visible_at_every_pixel_ratio(pixel_ratio):
    """Why this file measures a translucent line and not an opaque one.

    The opaque seam is a resampling coincidence: it is invisible at pixel ratio
    1 and only reaches 224/255 at 2. The translucent one is there at all of
    them, so a test built on it cannot pass by accident.
    """
    low, high = interior_range(render_corner(60, pixel_ratio=pixel_ratio))
    assert high - low > 3 * TOLERANCE, f"pixel ratio {pixel_ratio}: {low}..{high}"
