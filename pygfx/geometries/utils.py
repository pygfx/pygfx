import numpy as np


def merge(groups):
    positions = np.concatenate([g[0] for g in groups])
    normals = np.concatenate([g[1] for g in groups])
    texcoords = np.concatenate([g[2] for g in groups])
    indices = np.concatenate([g[3] for g in groups])
    i = 0
    j = 0
    for g in groups:
        indices[i:] += j
        # advance cursor to start of next group index
        i += len(g[3])
        j = len(g[0])
    return positions, normals, texcoords, indices
