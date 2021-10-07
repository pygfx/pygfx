import logging

import numpy as np


logger = logging.getLogger("pygfx")
logger.setLevel(logging.WARNING)


def array_from_shadertype(shadertype):
    """Get a numpy array object from a dict shadertype."""
    assert isinstance(shadertype, dict)

    primitives = {
        "i1": "int8",
        "u1": "uint8",
        "i2": "int16",
        "u2": "uint16",
        "i4": "int32",
        "u4": "uint32",
        "f2": "float16",
        "f4": "float32",
    }

    # Unravel the dict
    array_names = []
    dtype_fields = []
    for name, format in shadertype.items():
        primitive = primitives[format[-2:]]
        shapestr = format[:-2].replace("*", "x")
        shape = tuple(int(i) for i in shapestr.split("x") if i)
        dtype_fields.append((name, primitive, shape))
        if "*" in format:
            array_names.append(name)

    # Add meta field (zero bytes)
    # This isn't particularly pretty, but this way our metadata is attached
    # to the dtype without affecting its size.
    array_names.insert(0, "")
    array_names.append("")
    dtype_fields.append(("__".join(array_names), "uint8", (0,)))

    # Create a scalar of this type
    uniform_data = np.zeros((), dtype=dtype_fields)
    return uniform_data


def normals_from_vertices(rr, tris):
    """Efficiently compute vertex normals for a triangulated surface."""
    # This code was taken from Vispy
    # ensure highest precision for our summation/vectorization "trick"
    rr = rr[:, :3].astype(np.float64)
    tris = tris.reshape(-1, 3)
    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = np.cross((r2 - r1), (r3 - r1))

    # Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    npts = len(rr)

    # the following code replaces this, but is faster (vectorized):
    #
    # for p, verts in enumerate(tris):
    #     nn[verts, :] += tri_nn[p, :]
    #
    nn = np.zeros((npts, 3))
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(
                verts.astype(np.int32), tri_nn[:, idx], minlength=npts
            )
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]
    return nn.astype(np.float32)
