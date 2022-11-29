import pygfx as gfx
import pytest
from wgpu.gui.offscreen import WgpuCanvas


def test_repeated_show():
    pytest.xfail("Offscreen canvas can currently not be closed.")

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
