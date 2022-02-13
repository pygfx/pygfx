"""
Test that the examples run without error.
"""

import os
from pathlib import Path
import importlib
from unittest.mock import patch
import subprocess
import sys

import imageio
import numpy as np
import pytest


def _determine_lavapipe():
    code = "import wgpu.utils; d = wgpu.utils.get_default_device(); print(d.adapter.properties['adapterType'], d.adapter.properties['backendType'])"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return (
        result.stdout.strip().endswith("CPU Vulkan")
        and "traceback" not in result.stderr.lower()
    )


is_lavapipe = _determine_lavapipe()


ROOT = Path(__file__).parent.parent
examples_dir = ROOT / "examples"
screenshots_dir = examples_dir / "screenshots"
diffs_dir = screenshots_dir / "diffs"

# find examples that contain the marker comment for inclusion in the test suite
MARKER_COMMENT = "# test_example = true"
examples_to_test = []
for example_path in examples_dir.glob("*.py"):
    example_code = example_path.read_text()
    if MARKER_COMMENT in example_code:
        examples_to_test.append(example_path.stem)


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


@pytest.mark.parametrize("module", examples_to_test)
def test_examples(module, pytestconfig):
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
