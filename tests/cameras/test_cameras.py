import numpy as np
import pygfx as gfx
import pylinalg as pla


def test_otho_camera_near_far():
    for near, far in [
        (0, 100),
        (0, 10000),
        (-1200, 1200),
        (-40, 300),
        (-400, 0),
        (-400, -300),
        (300, 500),
    ]:
        camera = gfx.OrthographicCamera(20, 20, depth_range=(near, far))
        camera.update_projection_matrix()
        assert camera.near == near
        assert camera.far == far
        _run_for_camera(camera, near, far, True)


def test_perspective_camera_near_far():
    for near, far in [
        (0.1, 100),
        (0.1, 10000),
        (10, 1200),
        (200, 300),
        (490, 500),
    ]:
        camera = gfx.PerspectiveCamera(50, 1, depth_range=(near, far))
        camera.update_projection_matrix()
        assert camera.near == near
        assert camera.far == far
        _run_for_camera(camera, near, far, False)


def test_generic_camera_change_aspect():
    camera = gfx.PerspectiveCamera(0)
    camera._width = 100
    camera._height = 200

    assert camera.aspect == 0.5
    assert camera.width == 100
    assert camera.height == 200

    camera.aspect = 2

    assert camera.width == 200
    assert camera.height == 100

    camera.aspect = 1

    assert camera.width == 150
    assert camera.height == 150


def test_camera_show_methods():
    camera = gfx.PerspectiveCamera(0)

    # Show position
    camera.show_pos((100, 0, 0))
    assert camera.width == 100
    assert camera.height == 100
    assert np.allclose(camera.local.position, [0, 0, 0])

    # Show sphere with radius 200
    camera.show_object((0, 0, 0, 200), view_dir=(0, 0, -1))
    assert camera.width == 400
    assert camera.height == 400
    assert np.allclose(camera.local.position, [0, 0, 400])

    camera.local.rotation = (0, 0, 0, 1)  # reset rotation

    # Show rectangle
    camera.show_rect(0, 500, 0, 600, view_dir=(0, 0, -1))
    assert camera.width == 500
    assert camera.height == 600
    assert np.allclose(camera.local.position, [250, 300, 550])


def _run_for_camera(camera, near, far, check_halfway):
    # Some notes:
    #
    # * We use positions with negative z, because NDC looks down the
    #   negative Z axis. In other words, 1..-1 NDC maps to 0..1 in the
    #   depth buffer.
    # * Also note how with the ortho camera, a value in between the near
    #   and far plane results in 0.5 in the depth buffer, while for the
    #   perspective camera it'exponentiol, meaning that z-precision gets
    #   less precise further away.

    # This is not quite it ...
    # t1 = np.array(camera.projection_matrix.elements).reshape(4, 4).T
    # t2 = np.array(camera.projection_matrix_inverse.elements).reshape(4, 4)
    # pos_ndc1 = t1 @ np.array([0, 0, near, 1])
    # pos_ndc2 = t1 @ np.array([0, 0, 0.5 * (near + far), 1])
    # pos_ndc3 = t1 @ np.array([0, 0, far, 1])

    pos_ndc1 = pla.vec_transform((0, 0, -near), camera.projection_matrix)
    pos_ndc2 = pla.vec_transform(
        (0, 0, -0.5 * (near + far)), camera.projection_matrix
    )
    pos_ndc3 = pla.vec_transform((0, 0, -far), camera.projection_matrix)

    print("------", camera)
    print(pos_ndc1)
    print(pos_ndc2)
    print(pos_ndc3)

    assert np.allclose(pos_ndc1, [0, 0, 0])
    assert np.allclose(pos_ndc3, [0, 0, 1])
    if check_halfway:
        assert np.allclose(pos_ndc2, [0, 0, 0.5])

    pos_world1 = pla.vec_transform((0, 0, 0), camera.projection_matrix_inverse)
    pos_world2 = pla.vec_transform((0, 0, 0.5), camera.projection_matrix_inverse)
    pos_world3 = pla.vec_transform((0, 0, 1), camera.projection_matrix_inverse)

    assert np.allclose(pos_world1, [0, 0, -near])
    assert np.allclose(pos_world3, [0, 0, -far])
    if check_halfway:
        assert np.allclose(pos_world2, [0, 0, -0.5 * (near + far)])


def test_frustum():
    unit_cube_corners = np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ]
    )

    camera = gfx.OrthographicCamera(2, 2, depth_range=(-1, 1))
    frustum_corners = camera.frustum

    assert np.allclose(frustum_corners, unit_cube_corners.reshape(2, 4, 3))

    expected_corners = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-2.0, -2.0, -2.0],
            [2.0, -2.0, -2.0],
            [2.0, 2.0, -2.0],
            [-2.0, 2.0, -2.0],
        ]
    )

    camera = gfx.PerspectiveCamera(fov=90.0, aspect=1, depth_range=(1, 2))
    frustum_corners = camera.frustum

    assert np.allclose(frustum_corners, expected_corners.reshape(2, 4, 3))
