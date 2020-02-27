from visvis2.material._base import Material, stdinfo_type
from .._wrappers import BufferWrapper

import numpy as np
import wgpu
import python_shader
from python_shader import vec4, Array

# todo: use a compute shader or instancing?
# https://wwwtyro.net/2019/11/18/instanced-lines.html


@python_shader.python2shader
def compute_shader(
    index: (python_shader.RES_INPUT, "GlobalInvocationId", "i32"),
    pos1: (python_shader.RES_BUFFER, (1, 0), Array(vec4)),
    pos2: (python_shader.RES_BUFFER, (1, 1), Array(vec4)),
):
    p = pos1[index] * 1.0
    pos2[index * 2 + 0] = vec4(p.x, p.y + 5.0, p.z, 1.0)
    pos2[index * 2 + 1] = vec4(p.x, p.y - 5.0, p.z, 1.0)


@python_shader.python2shader
def vertex_shader(
    # io
    in_pos: (python_shader.RES_INPUT, 0, vec4),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
    # resources
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_type),
):
    world_pos = stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color: (python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


class LineStripMaterial(Material):
    def __init__(self):
        super().__init__()

        # Create buffer that only exist on the GPU, we provide stub data so that
        # the buffer knows type and strides (needed when used as vertex buffer)
        stub_array = np.zeros((0, 4), np.float32)
        self.positions2 = BufferWrapper(
            stub_array, nbytes=0, mapped=False, usage="vertex|storage"
        )

    def get_wgpu_info(self, wobject):
        # todo: we must hash the result by the len(geometry.position)
        geometry = wobject.geometry

        n = len(geometry.positions.data)  # number of vertices

        self.positions2.set_nbytes(2 * geometry.positions.nbytes)

        return [
            {
                "compute_shader": compute_shader,
                "indices": (n, 1, 1),
                "bindings1": [geometry.positions, self.positions2],
            },
            {
                "vertex_shader": vertex_shader,
                "fragment_shader": fragment_shader,
                "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
                "indices": n * 2,
                "vertex_buffers": [self.positions2],
                "target": None,  # default
            },
        ]
