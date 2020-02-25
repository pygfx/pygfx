from visvis2.material._base import Material, stdinfo_type
from .._wrappers import BufferWrapper

import wgpu
import python_shader
from python_shader import vec3, vec4, mat4, Array


# Idea:
# scene objects (like stdinfo, lights) are at bind group 0
# the objects defined by the geometry are at bind group 1
# the objects defined by the material are at bind group 2
# that leaves (by default) one bind group to be used for something else
# what about vertex data, put all vertices in storage buffers?


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
    index: (python_shader.RES_INPUT, "VertexId", "i32"),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
    # resources
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_type),
    positions: (python_shader.RES_BUFFER, (1, 1), Array(vec4)),
):
    pos3 = positions[index].xyz
    world_pos = stdinfo.world_transform * vec4(pos3, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color: (python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


class LineStripMaterial(Material):
    def __init__(self):
        super().__init__()
        self.positions2 = BufferWrapper(nbytes=0)

    #     self.uniforms = None
    #
    #     self.bindings = {0: (192, False)}
    #
    #
    #     self.shaders = {
    #         "compute": compute_shader,
    #         "vertex": vertex_shader,
    #         "fragment": fragment_shader,
    #     }
    #
    #     # self.primitive_topology = wgpu.PrimitiveTopology.line_strip
    #     self.primitive_topology = wgpu.PrimitiveTopology.triangle_strip

    def get_wgpu_info(self, obj):
        # todo: we must hash the result by the len(geometry.position)
        geometry = obj.geometry

        n = len(geometry.positions.data)  # number of vertices

        self.positions2.set_nbytes(2 * n * 4 * 4)

        return [
            {
                "compute_shader": compute_shader,
                "indices": (range(n), range(1), range(1)),
                "bindings1": [geometry.positions, self.positions2],
            },
            {
                "vertex_shader": vertex_shader,
                "fragment_shader": fragment_shader,
                "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
                "indices": range(n * 2),
                "target": None,  # default
                "bindings1": [geometry.positions, self.positions2],
            },
        ]
