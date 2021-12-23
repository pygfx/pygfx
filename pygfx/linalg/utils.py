import itertools

import numpy as np


MACHINE_EPSILON = (
    7.0 / 3 - 4.0 / 3 - 1
)  # the difference between 1 and the smallest floating point number greater than 1


def clamp(x: float, left: float, right: float) -> float:
    return max(left, min(right, x))


def transform(vectors, matrix, directions=False):
    shape = np.array(vectors.shape)
    shape[-1] += 1
    vectors_4d = np.empty(shape, dtype=vectors.dtype)
    vectors_4d[..., :-1] = vectors
    # if directions=True translation components of transforms will be ignored
    vectors_4d[..., -1] = 0 if directions else 1
    # ensure type conditions are met so we can use np.dot(out=) kwarg for performance
    matrix = matrix.astype(vectors_4d.dtype, copy=False).T
    return np.dot(vectors_4d, matrix, out=vectors_4d)[..., :-1]


def aabb_to_sphere(aabb):
    diagonal = aabb[1] - aabb[0]
    center = aabb[0] + diagonal / 2
    radius = np.linalg.norm(diagonal) / 2
    return np.array([*center, radius])


def transform_aabb(aabb, matrix):
    corners = np.array(list(itertools.product(*aabb.T)))
    corners_world = transform(corners, matrix)
    return np.array([corners_world.min(axis=0), corners_world.max(axis=0)])


def transform_sphere(sphere, matrix):
    center = transform(sphere[:3], matrix)
    radius = np.linalg.norm(transform(np.array([sphere[3], 0, 0]), matrix))
    return np.array([*center, radius])
