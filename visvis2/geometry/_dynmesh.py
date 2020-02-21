from ._base import Geometry


# I think this is where this thing goes, right?


class DynamicMeshGeometry(Geometry):
    """ A datasource, euh, I mean Geometry, specifically for
    triangulated meshes, supporting resampling and a variety of methods
    to query the mesh and make changes to it.
    """
