from python_shader import Struct, vec2, mat4
import numpy as np

from .._wrappers import BufferWrapper


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

        The default implementation returns a single render pipeline description,
        using vertex_shader, fragment_shader, and primitive_topology from the material,
        and index and vertex buffers from the geometry.

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

        # Get stuff from material

        vertex_shader = self.vertex_shader
        fragment_shader = self.fragment_shader
        primitive_topology = self.primitive_topology

        # Get stuff from geometry

        geometry = obj.geometry

        # Use index buffer if present on the geometry
        index_buffer = getattr(geometry, "index", None)
        index_buffer = index_buffer if isinstance(index_buffer, BufferWrapper) else None

        # All buffer objects are considered vertex buffers
        vertex_buffers = [
            val
            for val in geometry.__dict__.values()
            if isinstance(val, BufferWrapper) and val is not index_buffer
        ]
        if not vertex_buffers:
            raise ValueError("Cannot get default wgpu_info: no vertex buffers found.")

        if index_buffer:
            n = len(index_buffer.data)
        else:
            n = len(vertex_buffers[0].data)

        # Put it together!

        return [
            {
                "vertex_shader": vertex_shader,
                "fragment_shader": fragment_shader,
                "primitive_topology": primitive_topology,
                "indices": (range(n), range(1)),
                "index_buffer": index_buffer,
                "vertex_buffers": vertex_buffers,
            }
        ]
