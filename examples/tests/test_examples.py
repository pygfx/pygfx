"""
Test that the examples run without error.
"""

import os
import importlib
import runpy
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import psutil
from pympler import tracker
import pytest

from examples.tests.testutils import (
    is_lavapipe,
    find_examples,
    ROOT,
    screenshots_dir,
    diffs_dir,
)


# run all tests unless they opt-out
examples_to_run = find_examples(
    negative_query="# run_example = false", return_stems=True
)

# only test output of examples that opt-in
examples_to_test = find_examples(query="# test_example = true", return_stems=True)


count = 0
limit = 20
tr = None


@pytest.mark.parametrize("module", examples_to_run)
def test_examples_run(module):
    """Run every example marked to see if they can run without error."""
    global count, limit, tr

    count += 1
    if count >= limit:
        pytest.skip()
    
    if tr is None:
        tr = tracker.SummaryTracker()

    print("")
    mem_stats = psutil.virtual_memory()
    print(f"used system memory, before test start: {format_bytes(mem_stats.used)}")
    cpu_stats = psutil.cpu_percent()
    print(f"cpu freq, before test start: {cpu_stats}")

    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

    try:
        runpy.run_module(f"examples.{module}", run_name="__main__")
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]

        print("object diff, after test finish:")
        tr.print_diff()


@pytest.fixture
def force_offscreen():
    """Force the offscreen canvas to be selected by the auto gui module."""
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


@pytest.fixture
def mock_time():
    """Some examples use time to animate. Fix the return value
    for repeatable output."""
    with patch("time.time") as time_mock:
        time_mock.return_value = 1.23456
        yield


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    while size > power:
        size /= power
        n += 1
    label = power_labels[n] + "B"
    return f"{size:.2f} {label}"


@pytest.mark.parametrize("module", examples_to_test)
def test_examples_screenshots(module, pytestconfig, force_offscreen, mock_time):
    """Run every example marked for testing."""

    # render
    example = importlib.import_module(f"examples.{module}")
    img = example.renderer.target.draw()

    # check if _something_ was rendered
    assert img is not None and img.size > 0

    # we skip the rest of the test if you are not using lavapipe
    # images come out subtly differently when using different wgpu adapters
    # so for now we only compare screenshots generated with the same adapter (lavapipe)
    # a benefit of using pytest.skip is that you are still running
    # the first part of the test everywhere else; ensuring that examples
    # can at least import, run and render something
    if not is_lavapipe:
        pytest.skip("screenshot comparisons are only done when using lavapipe")

    # regenerate screenshot if requested
    screenshot_path = screenshots_dir / f"{module}.png"
    if pytestconfig.getoption("regenerate_screenshots"):
        imageio.imwrite(screenshot_path, img)

    # if a reference screenshot exists, assert it is equal
    assert (
        screenshot_path.exists()
    ), "found # test_example = true but no reference screenshot available"
    stored_img = imageio.imread(screenshot_path)
    # assert similarity
    is_similar = np.allclose(img, stored_img, atol=1)
    update_diffs(module, is_similar, img, stored_img)
    assert is_similar, (
        f"rendered image for example {module} changed, see "
        f"the {diffs_dir.relative_to(ROOT).as_posix()} folder"
        " for visual diffs (you can download this folder from"
        " CI build artifacts as well)"
    )


def update_diffs(module, is_similar, img, stored_img):
    diffs_dir.mkdir(exist_ok=True)

    diffs_rgba = None

    def get_diffs_rgba(slicer):
        # lazily get and cache the diff computation
        nonlocal diffs_rgba
        if diffs_rgba is None:
            # cast to float32 to avoid overflow
            # compute absolute per-pixel difference
            diffs_rgba = np.abs(stored_img.astype("f4") - img)
            # magnify small values, making it easier to spot small errors
            diffs_rgba = ((diffs_rgba / 255) ** 0.25) * 255
            # cast back to uint8
            diffs_rgba = diffs_rgba.astype("u1")
        return diffs_rgba[..., slicer]

    # split into an rgb and an alpha diff
    diffs = {
        diffs_dir / f"{module}-rgb.png": slice(0, 3),
        diffs_dir / f"{module}-alpha.png": 3,
    }

    for path, slicer in diffs.items():
        if not is_similar:
            diff = get_diffs_rgba(slicer)
            imageio.imwrite(path, diff)
        elif path.exists():
            path.unlink()


if __name__ == "__main__":
    # Enable tweaking in an IDE by running in an interactive session.
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    pytest.getoption = lambda x: False
    is_lavapipe = True  # noqa: F811
    test_examples_screenshots("validate_volume", pytest, None, None)
