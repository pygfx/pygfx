import numpy as np


def merge(groups):
    positions = np.concatenate([g[0] for g in groups])
    normals = np.concatenate([g[1] for g in groups])
    texcoords = np.concatenate([g[2] for g in groups])
    index = np.concatenate([g[3] for g in groups])
    i = 0
    j = 0
    for g in groups:
        index[i:] += j
        # advance cursor to start of next group index
        i += len(g[3])
        j = len(g[0])
    return positions, normals, texcoords, index


def transform(vectors, matrix, directions=False):
    shape = np.array(vectors.shape)
    shape[-1] += 1
    vectors_4d = np.empty(shape, dtype=vectors.dtype)
    vectors_4d[..., :-1] = vectors
    # if directions=True translation components of transforms will be ignored
    vectors_4d[..., -1] = 0 if directions else 1
    return np.dot(vectors_4d, matrix.T, out=vectors_4d)[..., :-1]
