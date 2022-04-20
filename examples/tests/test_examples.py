"""
Test that the examples run without error.
"""

import os
import importlib
from unittest.mock import patch
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
import pytest

from examples.tests.testutils import (
    is_lavapipe,
    find_examples,
    ROOT,
    screenshots_dir,
    diffs_dir,
)


# run all tests unless they opt-out
examples_to_run = find_examples(negative_query="# run_example = false")

# only test output of examples that opt-in
examples_to_test = find_examples(query="# test_example = true", return_stems=True)


@pytest.mark.parametrize("module", examples_to_run, ids=lambda module: module.stem)
def test_examples_run(module, pytestconfig):
    """Run every example marked to see if they can run without error."""
    env = os.environ.copy()
    env["WGPU_FORCE_OFFSCREEN"] = "true"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(module.relative_to(ROOT)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=ROOT,
            timeout=5,
            env=env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "opt-out by adding `# run_example = false` to the module docstring,"
            "or use WgpuAutoGui to support WGPU_FORCE_OFFSCREEN"
        )

    assert result.returncode == 0, f"failed to run:\n{result.stdout}"


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
    # cast to float32 to avoid overflow
    # compute absolute per-pixel difference
    diffs_rgba = np.abs(stored_img.astype("f4") - img)
    # magnify small values, making it easier to spot small errors
    diffs_rgba = ((diffs_rgba / 255) ** 0.25) * 255
    # cast back to uint8
    diffs_rgba = diffs_rgba.astype("u1")
    # split into an rgb and an alpha diff
    diffs = {
        diffs_dir / f"{module}-rgb.png": diffs_rgba[..., :3],
        diffs_dir / f"{module}-alpha.png": diffs_rgba[..., 3],
    }

    for path, diff in diffs.items():
        if not is_similar:
            imageio.imwrite(path, diff)
        elif path.exists():
            path.unlink()


if __name__ == "__main__":
    # Enable tweaking in an IDE by running in an interactive session.
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    pytest.getoption = lambda x: False
    is_lavapipe = True  # noqa: F811
    test_examples_screenshots("validate_volume", pytest, None, None)
