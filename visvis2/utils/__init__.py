from pyshader import shadertype_as_ctype
import numpy as np


def array_from_shadertype(shadertype):
    """ Get a numpy array object from a shadertype (from pyshader).
    """
    ctype = shadertype_as_ctype(shadertype)
    return np.asarray(ctype())
