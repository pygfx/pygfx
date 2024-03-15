"""Test that data can be accessed.
"""

import os
from pygfx.utils import get_resources_dir
from pygfx.renderers.wgpu import load_wgsl


def test_can_load_wgsl():
    wgsl = load_wgsl("line.wgsl")
    assert "fn vs_main(" in wgsl


def test_can_load_fonts():
    filename = os.path.join(get_resources_dir(), "NotoSans-Regular.ttf")
    assert os.path.isfile(filename)
