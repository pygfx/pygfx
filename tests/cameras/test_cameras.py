import numpy as np
import pygfx as gfx


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
        camera = gfx.OrthographicCamera(20, 20, near, far)
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
        camera = gfx.PerspectiveCamera(50, 1, near, far)
        camera.update_projection_matrix()
        assert camera.near == near
        assert camera.far == far
        _run_for_camera(camera, near, far, False)


def test_generic_camera_change_extend():
    camera = gfx.GenericCamera(0)
    camera._width = 100
    camera._height = 200

    assert camera.extent == 150
    assert camera.aspect == 0.5

    camera.extent *= 2

    assert camera.extent == 300
    assert camera.aspect == 0.5

    camera.extent /= 2

    assert camera._width == 100
    assert camera._height == 200


def test_generic_camera_change_aspect():
    camera = gfx.GenericCamera(0)
    camera._width = 100
    camera._height = 200

    assert camera.extent == 150
    assert camera.aspect == 0.5

    camera.aspect = 2

    assert camera.extent == 150
    assert camera._width == 200
    assert camera._height == 100

    camera.aspect = 1

    assert camera.extent == 150
    assert camera._width == 150
    assert camera._height == 150


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

    pos_ndc1 = gfx.linalg.Vector3(0, 0, -near).apply_matrix4(camera.projection_matrix)
    pos_ndc2 = gfx.linalg.Vector3(0, 0, -0.5 * (near + far)).apply_matrix4(
        camera.projection_matrix
    )
    pos_ndc3 = gfx.linalg.Vector3(0, 0, -far).apply_matrix4(camera.projection_matrix)

    print("------", camera)
    print(pos_ndc1)
    print(pos_ndc2)
    print(pos_ndc3)

    assert np.allclose(pos_ndc1.to_array(), [0, 0, 0])
    assert np.allclose(pos_ndc3.to_array(), [0, 0, 1])
    if check_halfway:
        assert np.allclose(pos_ndc2.to_array(), [0, 0, 0.5])

    pos_world1 = gfx.linalg.Vector3(0, 0, 0).apply_matrix4(
        camera.projection_matrix_inverse
    )
    pos_world2 = gfx.linalg.Vector3(0, 0, 0.5).apply_matrix4(
        camera.projection_matrix_inverse
    )
    pos_world3 = gfx.linalg.Vector3(0, 0, 1).apply_matrix4(
        camera.projection_matrix_inverse
    )

    assert np.allclose(pos_world1.to_array(), [0, 0, -near])
    assert np.allclose(pos_world3.to_array(), [0, 0, -far])
    if check_halfway:
        assert np.allclose(pos_world2.to_array(), [0, 0, -0.5 * (near + far)])


if __name__ == "__main__":
    test_otho_camera_near_far()
    test_perspective_camera_near_far()
    test_generic_camera_change_extend()
    test_generic_camera_change_aspect()
