from python_shader import Struct, vec2, mat4


# Definition of struct with standard info,
# provided to each world-object as uniform at slot 0.
stdinfo_type = Struct(world_transform=mat4, projection_transform=mat4, physical_size=vec2, logical_size=vec2)


class Material:
    def __init__(self):
        self.dirty = True
        self.shaders = {}
        self.uniforms = None
        self.primitive_topology = None

