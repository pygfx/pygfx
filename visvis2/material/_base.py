from python_shader import Struct, vec2, mat4
import numpy as np


# Definition of struct with standard info,
# provided to each world-object as uniform at slot 0.
stdinfo_type = Struct(
    world_transform=mat4,
    cam_transform=mat4,
    projection_transform=mat4,
    physical_size=vec2,
    logical_size=vec2,
)


def array_from_shader_type(spirv_type):
    """ Get a numpy array object from a SpirV type from python-shader.
    """
    return np.asarray(spirv_type())


class Material:
    def __init__(self):
        self.dirty = True
