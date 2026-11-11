import numpy as np
import pygfx as gfx
import pytest
import wgpu
from pygfx.utils.compute import ComputeShader

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_set_resource():
    device = gfx.renderers.wgpu.get_shared().device
    shader = ComputeShader(
        """
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            data[id.x] = 1.0;
        }
        """
    )
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    first_buffer = device.create_buffer(size=16, usage=usage)
    second_buffer = device.create_buffer(size=16, usage=usage)

    shader.set_resource(0, first_buffer)
    shader.dispatch(4)
    assert not shader.changed

    shader.set_resource(0, None)
    assert shader.changed

    shader.set_resource(0, second_buffer)
    shader.dispatch(4)
    data = np.frombuffer(device.queue.read_buffer(second_buffer), np.float32)
    assert data.tolist() == [1, 1, 1, 1]

    texture = gfx.Texture(np.zeros((4, 4), np.float32), dim=2)
    with pytest.raises(ValueError):
        shader.set_resource(0, texture, clear=True)
