from ._mesh import MeshBasicMaterial


class MeshVolumeSliceMaterial(MeshBasicMaterial):
    """ A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """
