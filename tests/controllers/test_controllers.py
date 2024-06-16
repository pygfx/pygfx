import pytest

from wgpu.gui.offscreen import WgpuCanvas
import pygfx as gfx


@pytest.mark.parametrize(
    "ControllerCls", (gfx.PanZoomController, gfx.FlyController, gfx.OrbitController, gfx.TrackballController)
)
def test_make_controller(ControllerCls):
    canvas = WgpuCanvas()
    renderer = gfx.WgpuRenderer(canvas)

    camera = gfx.PerspectiveCamera(width=100, height=100)

    controller = ControllerCls(camera, register_events=renderer)

    assert controller.cameras[0] is camera


@pytest.mark.parametrize(
    "ControllerCls", (gfx.PanZoomController, gfx.FlyController, gfx.OrbitController, gfx.TrackballController)
)
def test_add_remove_camera(ControllerCls):
    canvas = WgpuCanvas()
    renderer = gfx.WgpuRenderer(canvas)

    camera = gfx.PerspectiveCamera(width=100, height=100)

    controller = ControllerCls(camera, register_events=renderer)

    assert controller.cameras[0] is camera

    camera2 = gfx.PerspectiveCamera(width=100, height=100)
    camera3 = gfx.PerspectiveCamera(width=100, height=100)

    controller.add_camera(camera2, exclude_state={"x"})
    controller.add_camera(camera3, exclude_state={"y"})

    assert controller.cameras[1] is camera2
    assert controller.cameras[2] is camera3

    controller.remove_camera(camera2)

    assert controller.cameras[0] is camera
    assert controller.cameras[1] is camera3
