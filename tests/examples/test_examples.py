"""
Test that the examples run without error.
"""

import os
import importlib
from unittest.mock import patch
import subprocess
import sys

import imageio
import numpy as np
import pytest

from .testutils import is_lavapipe, find_examples, ROOT, screenshots_dir, diffs_dir


# run all tests unless they opt-out
examples_to_run = find_examples(negative_query="# run_example = false")

# only test output of examples that opt-in
examples_to_test = find_examples(query="# test_example = true", return_stems=True)


@pytest.fixture(autouse=True, scope="module")
def force_offscreen():
    """Force the offscreen canvas to be selected by the auto gui module."""
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    try:
        yield
    finally:
        del os.environ["WGPU_FORCE_OFFSCREEN"]


@pytest.fixture(autouse=True)
def mock_time():
    """Some examples use time to animate. Fix the return value
    for repeatable output."""
    with patch("time.time") as time_mock:
        time_mock.return_value = 1.23456
        yield


@pytest.mark.parametrize("module", examples_to_run, ids=lambda module: module.stem)
def test_examples_run(module, pytestconfig):
    """Run every example marked to see if they can run without error."""
    if not pytestconfig.getoption("slow"):
        pytest.skip("pass --slow to run")

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
        pytest.xfail(
            "opt-out by adding `# run_example = false` to the module docstring,"
            "or use WgpuAutoGui to support WGPU_FORCE_OFFSCREEN"
        )

    zero_exit = result.returncode == 0

    if not zero_exit:
        # in some cases it's pretty hard to support an example to run on CI
        # we xfail them on CI, but still allow them to run locally
        # would be nice to implement support later of course
        if "This application failed to start because no Qt platform plugin could be initialized." in result.stdout:
            pytest.xfail("Qt examples not supported on CI")
        if "ModuleNotFoundError: No module named 'wx'" in result.stdout:
            pytest.xfail("wx library is not available")

    assert zero_exit, f"failed to run:\n{result.stdout}"


@pytest.mark.parametrize("module", examples_to_test)
def test_examples_rendering(module, pytestconfig):
    """Run every example marked for testing."""

    # render
    example = importlib.import_module(f"examples.{module}")
    img = example.canvas.draw()

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
    if screenshot_path.exists():
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
    diffs = {
        diffs_dir / f"{module}-rgb.png": np.abs(stored_img[..., :3] - img[..., :3]),
        diffs_dir / f"{module}-alpha.png": np.abs(stored_img[..., 3] - img[..., 3]),
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
    test_examples("validate_volume", pytest)
