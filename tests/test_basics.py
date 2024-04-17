import sys
import subprocess

import wgpu
import pytest
import pygfx as gfx

from .testutils import can_use_wgpu_lib


def test_shared_object_not_created_at_import():
    code = "import pygfx; print(pygfx.renderers.wgpu.engine.shared.Shared._instance)"
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    result = p.stdout.strip()

    print(result)
    assert result.endswith("None")
    # If this fails, either the test is broken, or pygfx calls get_shared()
    # at import-time somewhere, which we don't want to do.


def test_shared_object():
    if not can_use_wgpu_lib:
        pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)

    get_shared = gfx.renderers.wgpu.get_shared

    assert get_shared() is not None
    assert isinstance(get_shared().adapter, wgpu.GPUAdapter)
    assert isinstance(get_shared().device, wgpu.GPUDevice)

    # Check that multiple invocations produce the same object - its a singleton
    assert get_shared() is get_shared()
