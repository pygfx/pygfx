import pytest
import wgpu
from wgpu.gui.offscreen import WgpuCanvas

import pygfx as gfx


def test_repeated_show():
    wgpu_version = tuple(int(x) for x in wgpu.__version__.split("."))

    if wgpu_version <= (0, 8, 2):
        pytest.xfail("Offscreen canvas can not be closed.")

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # always use offscreen canvas
    disp = gfx.Display(canvas=WgpuCanvas())

    disp.show(cube)

    with pytest.raises(RuntimeError):
        # repeated show should raise - currently doesn't because offscreen
        # canvases can't be closed
        disp.show(cube)
