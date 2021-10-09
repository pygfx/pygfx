import numpy as np
import pygfx as gfx


def test_compare_to_three():
    ref_positions = [
        [2, 3, 4, 2, 3, -4, 2, -3, 4, 2, -3, -4],
        [-2, 3, -4, -2, 3, 4, -2, -3, -4, -2, -3, 4],
        [-2, 3, -4, 2, 3, -4, -2, 3, 4, 2, 3, 4],
        [-2, -3, 4, 2, -3, 4, -2, -3, -4, 2, -3, -4],
        [-2, 3, 4, 2, 3, 4, -2, -3, 4, 2, -3, 4],
        [2, 3, -4, -2, 3, -4, 2, -3, -4, -2, -3, -4],
    ]
    ref_positions = np.array(ref_positions, np.float32).reshape((-1, 3))

    ref_normals = [
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1],
    ]
    ref_normals = np.array(ref_normals, np.float32).reshape((-1, 3))

    ref_texcoords = [
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    ]
    ref_texcoords = np.array(ref_texcoords, np.float32).reshape((-1, 2))

    ref_indices = [
        [0, 2, 1, 2, 3, 1, 4, 6, 5, 6, 7, 5, 8, 10, 9, 10, 11, 9],
        [12, 14, 13, 14, 15, 13, 16, 18, 17, 18, 19, 17, 20, 22, 21, 22, 23, 21],
    ]
    ref_indices = np.array(ref_indices, np.uint32).reshape((-1, 3))

    b = gfx.BoxGeometry(4, 6, 8)

    assert np.alltrue(b.positions.data[:, :3] == ref_positions)
    # assert np.alltrue(b.normals.data[:,:3] == ref_normals)
    assert np.alltrue(b.texcoords.data == ref_texcoords)
    assert np.alltrue(b.index.data == ref_indices)


if __name__ == "__main__":
    test_compare_to_three()
