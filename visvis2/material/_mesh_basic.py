from ._base import Material

# todo: put in an example somewhere how to use storage buffers for vertex data:
# index: (python_shader.RES_INPUT, "VertexId", "i32")
# positions: (python_shader.RES_BUFFER, (1, 0), "Array(vec4)")
# position = positions[index]


class MeshBasicMaterial(Material):
    def __init__(self):
        super().__init__()
