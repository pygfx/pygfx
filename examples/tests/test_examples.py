"""
Unit tests for our examples.

* Some meta tests.
* On a subset: test that examples run without error.
* On a smaller subset: test that examples produce the correct screenshot (compared to a reference).
"""

import os
import importlib
import logging
import runpy
import sys
from unittest.mock import patch

import imageio.v3 as iio
import numpy as np
import pytest

from .testutils import (
    adapter,
    is_lavapipe,
    find_examples,
    ROOT,
    screenshots_dir,
    diffs_dir,
)


examples_info = find_examples()  # (filename, fname, test_config, doc_config)
examples_to_run = [ex[0] for ex in examples_info if ex[2] == "run"]
examples_to_compare = [ex[0] for ex in examples_info if ex[2] == "compare"]


class LogHandler(logging.Handler):
    def __init__(self, *args):
        super().__init__(*args)
        self.records = []

    def emit(self, record):
        if record.name in ["trimesh", "imageio"]:
            return
        self.records.append(record)


log_handler = LogHandler(logging.WARN)
logging.getLogger().addHandler(log_handler)


def test_that_we_are_on_lavapipe():
    print(adapter.info)
    if os.getenv("PYGFX_EXPECT_LAVAPIPE"):
        assert is_lavapipe


def test_examples_meta():
    """Make sure every example has a proper test-config comment."""

    errors = []
    ok_configs = "off", "run", "compare"

    # Check configs
    per_stem = {}
    for filename, fname, test_config, docs_config in examples_info:
        per_stem.setdefault(filename.stem, []).append(fname)
        if test_config is None:
            errors.append(f"Example '{fname}' has missing test config.")
        elif test_config not in ok_configs:
            errors.append(f"Example '{fname}' has unexpected test config.")
        if docs_config is None:
            errors.append(f"Example '{fname}' has missing gallery config.")

    # Check that filenames are unique
    for _stem, fnames in per_stem.items():
        if len(fnames) != 1:
            errors.append(f"Name clash: {fnames}")

    assert not errors, "Meta-errors in examples:\n" + "\n".join(errors)


@pytest.mark.parametrize("filename", examples_to_run, ids=lambda x: x.stem)
def test_examples_run(filename, force_offscreen):
    """Run every example marked to see if they can run without error."""

    # use runpy so the module is not actually imported (and can be gc'd)
    # but also to be able to run the code in the __main__ block

    # Reset logged warnings/errors
    log_handler.records = []

    try:
        runpy.run_path(filename, run_name="__main__")
    except (ModuleNotFoundError, ImportError) as e:
        str_e = str(e)
        if str_e == "No module named 'trimesh'":
            pytest.skip("trimesh is not installed")
        elif (
            str_e
            == "The `gltflib` library is required to load gltf scene: pip install gltflib"
        ):
            pytest.skip("gltflib is not installed")
        elif (
            str_e
            == "The `trimesh` library is required to load meshes: pip install trimesh"
        ):
            pytest.skip("trimesh is not installed")
        else:
            raise e

    # If any errors occurred in the draw callback, they are logged
    if log_handler.records:
        raise RuntimeError("Example generated errors during draw")


@pytest.mark.parametrize("filename", examples_to_compare, ids=lambda x: x.stem)
def test_examples_compare(filename, pytestconfig, force_offscreen, mock_time):
    """Run every example marked to compare its result against a reference screenshot."""

    # import the example module
    module_name = filename.stem
    module = import_from_path(module_name, filename)

    # render a frame
    img = np.asarray(module.renderer.target.draw())

    # check if _something_ was rendered
    assert img is not None and img.size > 0

    # we skip the rest of the test if you are not using lavapipe
    # images come out subtly differently when using different wgpu adapters
    # so for now we only compare screenshots generated with the same adapter (lavapipe)
    # a benefit of using pytest.skip is that you are still running
    # the first part of the test everywhere else; ensuring that examples
    # can at least import, run and render something
    if not is_lavapipe:
        pytest.skip(
            "screenshot comparisons are only done when using lavapipe. "
            "Rerun with PYGFX_WGPU_ADAPTER_NAME=llvmpipe"
        )

    # regenerate screenshot if requested
    screenshot_path = screenshots_dir / f"{module_name}.png"
    if pytestconfig.getoption("regenerate_screenshots"):
        iio.imwrite(screenshot_path, img)

    # if a reference screenshot exists, assert it is equal
    assert screenshot_path.exists(), "found no reference screenshot"
    stored_img = iio.imread(screenshot_path)

    # assert similarity
    atol = 1
    try:
        np.testing.assert_allclose(img, stored_img, atol=atol)
        is_similar = True
    except Exception as e:
        is_similar = False
        raise AssertionError(
            f"rendered image for example {module_name} changed, see "
            f"the {diffs_dir.relative_to(ROOT).as_posix()} folder"
            " for visual diffs (you can download this folder from"
            " CI build artifacts as well)"
        ) from e
    finally:
        update_diffs(module_name, is_similar, img, stored_img, atol=atol)


def import_from_path(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # With this approach the module is not added to sys.modules, which
    # is great, because that way the gc can simply clean up when we lose
    # the reference to the module
    assert module.__name__ == module_name
    assert module_name not in sys.modules

    return module


def update_diffs(module, is_similar, img, stored_img, *, atol):
    diffs_dir.mkdir(exist_ok=True)

    if is_similar:
        for path in [
            # Keep filename in sync with the ones generated below
            diffs_dir / f"{module}-rgb.png",
            diffs_dir / f"{module}-alpha.png",
            diffs_dir / f"{module}-rgb-above_atol.png",
            diffs_dir / f"{module}-alpha-above_atol.png",
            diffs_dir / f"{module}.png",
        ]:
            if path.exists():
                path.unlink()
        return

    # cast to float32 to avoid overflow
    # compute absolute per-pixel difference
    diffs_rgba = np.abs(stored_img.astype("f4") - img)

    diffs_rgba_above_atol = diffs_rgba.copy()
    diffs_rgba_above_atol[diffs_rgba <= atol] = 0

    # magnify small values, making it easier to spot small errors
    diffs_rgba = ((diffs_rgba / 255) ** 0.25) * 255
    # cast back to uint8
    diffs_rgba = diffs_rgba.astype("u1")

    diffs_rgba_above_atol = ((diffs_rgba_above_atol / 255) ** 0.25) * 255
    diffs_rgba_above_atol = diffs_rgba_above_atol.astype("u1")

    # split into an rgb and an alpha diff
    # And highlight differences that are above the atol
    iio.imwrite(diffs_dir / f"{module}-rgb.png", diffs_rgba[..., :3])
    iio.imwrite(diffs_dir / f"{module}-alpha.png", diffs_rgba[..., 3])
    iio.imwrite(
        diffs_dir / f"{module}-rgb-above_atol.png", diffs_rgba_above_atol[..., :3]
    )
    iio.imwrite(
        diffs_dir / f"{module}-alpha-above_atol.png", diffs_rgba_above_atol[..., 3]
    )
    iio.imwrite(diffs_dir / f"{module}.png", img)


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


if __name__ == "__main__":
    # Enable tweaking in an IDE by running in an interactive session.
    os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
    pytest.getoption = lambda x: False
    test_examples_compare("validate_volume", pytest, None, None)
