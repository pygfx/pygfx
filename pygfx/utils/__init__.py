"""
Utility functions for Pygfx.

.. currentmodule:: pygfx.utils

.. autosummary::
    :toctree: utils/
    :template: ../_templates/custom_layout.rst

    Color
    load.load_mesh
    load.load_meshes
    load.load_scene
    show.show
    show.Display
    viewport.Viewport
    text.FontManager
    enums
    cm

Transform classes
-----------------

These classes are used internally to create an intuitive API to transform
WorldObjects (`obj.world` and `obj.local`). They are listed here for
completeness and you will likely never have to instantiate them directly.

.. autosummary::
    :toctree: utils/
    :template: ../_templates/custom_layout.rst

    transform.AffineBase
    transform.AffineTransform
    transform.RecursiveTransform

"""

import os
import types
import logging
import inspect

import numpy as np

from .color import Color  # noqa: F401
from . import enums  # noqa: F401

from ._dirs import get_resources_dir, get_cache_dir  # noqa: F401

logger = logging.getLogger("pygfx")


def _set_log_level():
    # Set default level
    logger.setLevel(logging.WARN)
    # Set user-specified level
    level = os.getenv("PYGFX_LOG_LEVEL", "")
    if level:
        try:
            if level.isnumeric():
                logger.setLevel(int(level))
            else:
                logger.setLevel(level.upper())
        except Exception:
            logger.warning(f"Invalid pygfx log level: {level}")


_set_log_level()


def array_from_shadertype(shadertype, count=None):
    """Get a numpy array object from a dict shadertype.

    The fields are re-ordered and padded as necessary to fulfil alignment rules.
    See https://www.w3.org/TR/WGSL/#structure-layout-rules

    A note on matrices (e.g. "4x4xf4" or "2x3xf4"): from the perspective of Python,
    the matrices in a uniform array are transposed, and nx3 matrices are actually nx4.

    To deal with this, setting a matrix in a uniform goes like this:

        uniform.data["a_matrix"] = numpy_array.T
        uniform.data["an_nx3_matrix"][:, :3] = numpy_array.T

    The reason is that WGSL matrices are column-major, while Numpy arrays are
    row-major (i.e. C-order) by default. Although numpy does support
    column-major arrays (F-order), it looks like we cannot apply this to the
    sub-arrays in the uniform struct. And for the nx3 matrices, WGSL's alignment
    constraint requires padding for each column in an nx3 matrix, so it takes up
    the same space as an nx4 matrix.

    params:
        shadertype: dict
            A dict containing the shadertype.
        count: None or int
            If count is not None, array has a shape of (count, ),
            Indicates that the corresponding "Buffer" is an array of struct.
    """
    assert isinstance(shadertype, dict)
    assert count is None or count > 0

    # Note: we have yet to add f2 (16 bit float)
    primitives = {
        "i4": "int32",
        "u4": "uint32",
        "f4": "float32",
    }

    class Field:
        __slots__ = [
            "align",
            "name",
            "primitive",
            "shape",
            "size",
        ]

        def __init__(self, name, format):
            self.name = name
            if format[-2:] not in primitives:
                raise RuntimeError(
                    f"Values in a uniform must have a 32bit primitive type, not {format}"
                )
            self.primitive = primitives[format[-2:]]
            primitive_size = 4  # no support for f2 yet
            # Get shape, excluding array part
            arraystr, _, shapestr = format[:-2].rpartition("*")
            shape = [int(i) for i in shapestr.split("x") if i]
            shape = shape or [1]
            # The alignment is based on the last element of te (original) shape
            self.align = shape[-1] * primitive_size
            # Snap align to factors of 2.
            # 3xf2 -> 6 -> 8
            # 3x3xf4 -> 12 -> 16
            for ref_align in [2, 4, 8, 16]:
                if self.align <= ref_align:
                    self.align = ref_align
                    break
            # Deal with nx3 matrices; they are nx4 internally
            if len(shape) > 1 and shape[-1] == 3:
                mat3_names.append(name)
                shape[-1] = 4
            # Include array size in shape
            if arraystr:
                array_names.append(name)
                shape.insert(0, int(arraystr))
            # Calculate size, the number of bytes that this field occupies
            self.size = int(np.prod(shape)) * primitive_size
            # Convert shape to tuple for the numpy sub array.
            # We do NOT reverse the shape, see function docstring.
            self.shape = () if (len(shape) == 1 and shape[0] == 1) else tuple(shape)

        def use(self):
            result = self.name, self.primitive, self.shape
            self.name += "-already used"  # break when used twice
            return result

    pad_index = 0
    array_names = []
    mat3_names = []
    fields_per_align = {16: [], 8: [], 4: [], 2: []}
    packed_fields = []  # what fields fill up space to fix alignment

    # The definition of the dtype of the structured array, as a list of tuples: (name, primitive-dtype, shape)
    # We could also use the dict flavor, setting explicit offsets for each field instead of stub padding fields,
    # but its also an advantage during debugging that the padding is explicit/visible.
    dtype_fields = []

    # The size/alignment of a struct is the max alignment of its members.
    # But when a uniform buffer is an array of structs, the striding of the array must be a multiple of 16 (std140-like rules).
    struct_alignment = 0
    if count is not None:
        struct_alignment = 16

    # Collect fields per alignment, so we can process from big to small to avoid padding.
    for name, format in shadertype.items():
        field = Field(name, format)
        fields_per_align[field.align].append(field)
        struct_alignment = max(struct_alignment, field.align)

    def fill_bytes(ref_align, nbytes):
        # This function only gets called when using a vec3<x32> or vec3<f16>. Since the
        # fields are sorted by alignment (big to small) this only applies to types that
        # align weirdly. But mot for the nx3 matrices, because they have intrinsic padding.
        nonlocal pad_index
        j = ref_align - nbytes
        # See if we can fill the gap with other fields
        for align in [4, 2]:
            if align >= ref_align or j % align > 0:
                continue
            fields = fields_per_align[align]
            checked_all = False
            while nbytes >= align and not checked_all:
                selected_field = None
                for field in fields:
                    if field.size <= nbytes:
                        selected_field = field
                        packed_fields.append(field.name)
                        dtype_fields.append(field.use())
                        nbytes -= field.size
                        j += field.size
                        break
                if selected_field:
                    fields.remove(selected_field)
                else:
                    checked_all = True
        # Fill remaining space with padding
        if nbytes:
            pad_index += 1
            dtype_fields.append((f"__padding{pad_index}", "uint8", (need_bytes,)))

    # Process the fields, from big to small
    i = 0  # bytes processed
    for ref_align in [16, 8, 4, 2]:
        for field in fields_per_align[ref_align]:
            too_many_bytes = i % field.align
            if too_many_bytes:
                need_bytes = field.align - too_many_bytes
                assert need_bytes < 16
                assert ref_align in (16, 8)
                fill_bytes(ref_align, need_bytes)
                i += need_bytes
            # Add the field itself
            dtype_fields.append(field.use())
            i += field.size

    # Add padding to the struct
    too_many_bytes = i % struct_alignment
    if too_many_bytes:
        need_bytes = struct_alignment - too_many_bytes
        pad_index += 1
        dtype_fields.append((f"__padding{pad_index}", "uint8", (need_bytes,)))
        i += need_bytes

    # Nice for debugging:
    # if packed_fields or pad_index > 1:
    #     print("fields:")
    #     for field in dtype_fields:
    #         print("   ", field)

    # Add meta fields (zero bytes)
    # This isn't particularly pretty, but this way our metadata is attached
    # to the dtype without affecting its size.
    if array_names:
        dtype_fields.append(
            ("__meta_array_names__" + "__".join(array_names), "uint8", (0,))
        )
    if mat3_names:
        dtype_fields.append(
            ("__meta_mat3_names__" + "__".join(mat3_names), "uint8", (0,))
        )

    # Create a scalar of this type
    if count is not None:
        uniform_data = np.zeros((count,), dtype=dtype_fields)
    else:
        uniform_data = np.zeros((), dtype=dtype_fields)

    # If this fails we did something wrong above
    assert uniform_data.nbytes % struct_alignment == 0

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


def assert_type(name, value, *classes):
    allow_none = False
    if classes[0] is None:
        if value is None:
            return
        allow_none = True
        classes = classes[1:]

    if not isinstance(value, classes):
        # Get traceback object to point of the frame of interest
        f = inspect.currentframe()
        f = f.f_back
        if name:
            # Step back to calling code
            f = f.f_back
            # If this is a constructor that has name as a (kw) argument, take another step back
            if f.f_code.co_name == "__init__" and name in f.f_code.co_varnames:
                f = f.f_back
        tb = types.TracebackType(None, f, f.f_lasti, f.f_lineno)

        # Build error message
        msg = "Expected"
        if name:
            msg += f" '{name}' to be"
        class_strings = [cls.__name__ for cls in classes]
        msg += f" an instance of {' | '.join(class_strings)}"
        if allow_none:
            msg += " or None"
        valuestr = value.__class__.__name__
        msg += f", but got {valuestr} object."

        # Raise message with alt traceback
        raise TypeError(msg).with_traceback(tb) from None


class ReadOnlyDict(dict):
    """A read-only dict, for storing structured data that can be hashed."""

    __slots__ = ["_hash"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Calculate hash in a way that requires any value to also be hashable
        parts = []
        for k in sorted(self.keys()):
            v = self[k]
            parts.append(str(hash(k)))
            parts.append(str(hash(v)))
        self._hash = hash(" ".join(parts))

    def __setitem__(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def __delitem__(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def clear(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def pop(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def popitem(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def setdefault(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def update(self, *args, **kwargs):
        raise TypeError("Cannot modify ReadOnlyDict")

    def __hash__(self):
        return self._hash
