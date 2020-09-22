from ._base import WorldObject
from ..geometries import BoxGeometry
from ..resources import Buffer


class Volume(WorldObject):
    """A volume represents a 3D image in space. It has an implicit
    geometry based on the shape of the texture (the map of the
    material), and the voxel (0,0,0) will be at the origin of the local
    coordinate frame. Positioning and dealing with anisotropy should
    be dealt with using the scale and position properties.
    """

    def __init__(self, material):
        super().__init__()
        self.material = material

    @property
    def material(self):
        """The material of the volume."""
        return self._material

    @material.setter
    def material(self, material):
        self._material = material
        self._set_geometry(material.map.size)

    def _set_geometry(self, size):
        # Create box geometry, and map to 0..1
        geometry = BoxGeometry(1, 1, 1)
        geometry.positions.data[:, :3] += 0.5
        # This is our 3D texture coords
        geometry.texcoords = Buffer(
            geometry.positions.data[:, :3].copy(), usage="vertex|storage"
        )
        # Map to volume size
        for i in range(3):
            geometry.positions.data[:, i] *= size[i]
        # Apply
        self.geometry = geometry
