import logging

import numpy as np

from .color import Color  # noqa: F401
from . import cm  # noqa: F401

logger = logging.getLogger("pygfx")
logger.setLevel(logging.WARNING)


def array_from_shadertype(shadertype):
    """Get a numpy array object from a dict shadertype."""
    assert isinstance(shadertype, dict)

    primitives = {
        "i4": "int32",
        "u4": "uint32",
        "f4": "float32",
    }

    # Unravel the dict, turning it into a numpy array.
    # We also sort the fields so that the uniforms are properly aligned.
    # See https://www.w3.org/TR/WGSL/#structure-layout-rules
    # Note that 3x4xf4 matches a mat3x4<f32>
    array_names = []
    dtype_fields = []
    for name, format in shadertype.items():
        if format[-2:] not in primitives:
            raise RuntimeError(
                f"Values in a uniform must have a 32bit primitive type, not {format}"
            )
        primitive = primitives[format[-2:]]
        # Get shape, excluding array part
        shapestr = format[:-2].split("*")[-1]
        shape = [int(i) for i in shapestr.split("x") if i]
        align_size = shape[-1] if shape else 1  # in mat2x4 we need the 4
        if align_size == 3:  # vec3 and matnx3 are forbidden for now
            raise ValueError(
                f"Uniform format {format} forbidden for now due to alignment."
            )
        shape.reverse()  # reverse because numpy is row-major
        # Include array size
        if "*" in format:
            array_names.append(name)
            shape.insert(0, int(format.split("*")[0]))
        # Create field, include align_size for sorting
        dtype_fields.append((name, primitive, tuple(shape), align_size))

    # Sort by alignment, then strip the align_size (helper element) from the tuple
    dtype_fields.sort(key=lambda field: -field[-1])
    dtype_fields = [field[:-1] for field in dtype_fields]

    # Add meta field (zero bytes)
    # This isn't particularly pretty, but this way our metadata is attached
    # to the dtype without affecting its size.
    array_names.insert(0, "")
    array_names.append("")
    dtype_fields.append(("__".join(array_names), "uint8", (0,)))

    # Add padding: uniform buffers must align to 16 bytes.
    size = np.dtype(dtype_fields).itemsize
    n16 = int(np.ceil(size / 16))
    padding = n16 * 16 - size
    dtype_fields.append(("__padding", "uint8", (padding,)))

    # Create a scalar of this type
    uniform_data = np.zeros((), dtype=dtype_fields)
    return uniform_data


def unpack_bitfield(packed, **bit_counts):
    """Unpack values from an uint64 bitfield."""
    values = {}
    for key, bits in bit_counts.items():
        mask = 2**bits - 1
        values[key] = packed & mask
        packed = packed >> bits
    return values


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
