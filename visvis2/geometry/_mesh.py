from python_shader import vec4, Array

from ._base import Geometry


# I think this is where this thing goes, right?


class MeshGeometry(Geometry):
    """ Geometry for a polygonial mesh.
    """

    # todo: we can use a definition like this to ensure that the shaders use the correct types
    # todo: but then we must also validate that any binding arrays that get set match up
    binding_def = {
        "positions": (0, Array(vec4)),
        "colors": (1, Array(vec4)),
    }
