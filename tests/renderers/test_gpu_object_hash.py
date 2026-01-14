"""Test for GPU object hash stability across garbage collection.

This tests that hash_from_value produces unique hashes for different GPU objects,
even when Python reuses memory addresses (id() values) after garbage collection.

Bug: The original implementation used id() (memory address) in the JSON encoding
of GPU objects. When an object was garbage collected and a new object was allocated
at the same memory address, the hash would incorrectly match the old object's hash,
causing cache collisions in any cache using hash_from_value().

Fix: Use a stable monotonically increasing counter with WeakKeyDictionary instead
of id() to ensure each GPU object gets a unique identifier that is never reused.
"""

import gc
import wgpu
import pytest

from pygfx.renderers.wgpu.engine.shared import get_shared
from pygfx.renderers.wgpu.engine.utils import jsonencoder


def test_gpu_object_json_encoding_unique():
    """Test that JSON encoding of GPU objects produces unique strings.

    This is a more direct test of the encoding mechanism. Each GPU object
    should produce a unique JSON representation.
    """
    device = get_shared().device

    seen_encodings = set()

    for i in range(50):
        buffer = device.create_buffer(size=256, usage=wgpu.BufferUsage.STORAGE)
        encoding = jsonencoder.encode(buffer)

        assert encoding not in seen_encodings, (
            f"Duplicate encoding detected: {encoding}. "
            f"Different GPU objects should have unique encodings."
        )
        seen_encodings.add(encoding)

        del buffer
        gc.collect()
