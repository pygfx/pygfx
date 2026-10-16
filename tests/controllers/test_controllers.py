import pytest

from rendercanvas.offscreen import RenderCanvas
import pygfx as gfx


@pytest.mark.parametrize(
    "controller_cls",
    (
        gfx.PanZoomController,
        gfx.FlyController,
        gfx.OrbitController,
        gfx.TrackballController,
    ),
)
def test_make_controller(controller_cls):
    canvas = RenderCanvas()
    renderer = gfx.WgpuRenderer(canvas)

    camera = gfx.PerspectiveCamera(width=100, height=100)

    controller = controller_cls(camera, register_events=renderer)

    assert controller.cameras[0] is camera


@pytest.mark.parametrize(
    "controller_cls",
    (
        gfx.PanZoomController,
        gfx.FlyController,
        gfx.OrbitController,
        gfx.TrackballController,
    ),
)
def test_add_remove_camera(controller_cls):
    canvas = RenderCanvas()
    renderer = gfx.WgpuRenderer(canvas)

    camera = gfx.PerspectiveCamera(width=100, height=100)

    controller = controller_cls(camera, register_events=renderer)

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


def make_panzoom():
    canvas = RenderCanvas(size=(200, 100))
    renderer = gfx.WgpuRenderer(canvas)
    camera = gfx.OrthographicCamera(width=200, height=100, maintain_aspect=False)
    controller = gfx.PanZoomController(camera, register_events=renderer)
    return camera, controller


def test_zoom_to_point_scalar_equals_uniform_tuple():
    rect = (0, 0, 200, 100)
    pos = (150, 25)

    camera1, controller1 = make_panzoom()
    controller1.zoom_to_point(1.0, pos, rect)

    camera2, controller2 = make_panzoom()
    controller2.zoom_to_point((1.0, 1.0), pos, rect)

    assert camera1.width == pytest.approx(100)
    assert camera1.height == pytest.approx(50)
    assert camera2.width == pytest.approx(camera1.width)
    assert camera2.height == pytest.approx(camera1.height)
    assert tuple(camera2.local.position) == pytest.approx(tuple(camera1.local.position))


def test_zoom_to_point_per_axis():
    rect = (0, 0, 200, 100)
    # pointer right of and above the center
    pos = (150, 25)

    camera, controller = make_panzoom()
    x0, y0, _ = camera.local.position

    # zoom in horizontally only
    controller.zoom_to_point((1.0, 0.0), pos, rect)

    assert camera.width == pytest.approx(100)
    assert camera.height == pytest.approx(100)
    # the view pans towards the pointer along x, and not at all along y
    assert camera.local.position[0] > x0
    assert camera.local.position[1] == pytest.approx(y0)

    # and vertically only
    controller.zoom_to_point((0.0, 1.0), pos, rect)

    assert camera.width == pytest.approx(100)
    assert camera.height == pytest.approx(50)
