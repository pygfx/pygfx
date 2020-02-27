from visvis2.material._base import Material, stdinfo_type

import wgpu  # only for flags/enums
import python_shader
from python_shader import vec4

# todo: put in an example somewhere how to use storage buffers for vertex data:
# index: (python_shader.RES_INPUT, "VertexId", "i32")
# positions: (python_shader.RES_BUFFER, (1, 0), "Array(vec4)")
# position = positions[index]


@python_shader.python2shader
def vertex_shader(
    # input and output
    position: (python_shader.RES_INPUT, 0, vec4),
    out_pos: (python_shader.RES_OUTPUT, "Position", vec4),
    # uniform and storage buffers
    stdinfo: (python_shader.RES_UNIFORM, (0, 0), stdinfo_type),
):
    world_pos = stdinfo.world_transform * vec4(position.xyz, 1.0)
    ndc_pos = stdinfo.projection_transform * stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader assign to input arg


@python_shader.python2shader
def fragment_shader(out_color: (python_shader.RES_OUTPUT, 0, vec4),):
    out_color = vec4(1.0, 0.0, 0.0, 1.0)  # noqa - shader assign to input arg


class MeshBasicMaterial(Material):
    def __init__(self):
        super().__init__()
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.primitive_topology = wgpu.PrimitiveTopology.triangle_list
