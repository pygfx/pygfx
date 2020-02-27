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

    def get_wgpu_info(self, obj):
        """ Return a list of high level pipeline descriptions.
        These can be compute or render pipelines.

        Example compute pipeline:

            {
                "compute_shader": shader_module,
                "indices": (n, 1, 1),  # the number of iterations (x, y, z)
                "bindings1": [buffer1, texture1],  # optional
                "bindings2": [buffer2],  # optional
            },

        Example render pipeline:

            {
                "vertex_shader": vertex_shader,
                "fragment_shader": fragment_shader,
                "primitive_topology": "triangle-strip",
                "indices": range(10, n),  # int or range
                "index_buffer": buffer3,  # optional
                "vertex_buffers": [buffer4],  # optional
                "target": texture2,  # optional, not-implemented
            },
        """
        raise NotImplementedError()
