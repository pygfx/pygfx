from ._base import WorldObject
from ..geometries import BoxGeometry
from ..resources import Buffer


class Volume(WorldObject):
    """A volume represents a 3D image in space. It has an implicit
    geometry based on the shape of the texture (the map of the
    material), and the center of voxel (0,0,0) will be at the origin
    of the local coordinate frame. Positioning and dealing with
    anisotropy should be dealt with using the scale and position
    properties.

    The picking info of a Volume (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``voxel_index`` (tuple of 3 floats).
    """

    def __init__(self, size, material):
        super().__init__()
        self.size = size
        self.material = material

    @property
    def material(self):
        """The material of the volume."""
        return self._material

    @material.setter
    def material(self, material):
        self._material = material

    @property
    def size(self):
        """The size of the volume (xyz)."""
        return self._size

    @size.setter
    def size(self, size):
        # Check and store
        x, y, z = size
        self._size = size = int(x), int(y), int(z)
        # Create box geometry, and map to 0..1
        geometry = BoxGeometry(1, 1, 1)
        geometry.positions.data[:, :3] += 0.5
        # This is our 3D texture coords
        geometry.texcoords = Buffer(
            geometry.positions.data[:, :3].copy(), usage="vertex|storage"
        )
        # Map to volume size
        for i in range(3):
            column = geometry.positions.data[:, i]
            column *= size[i]
            column -= 0.5  # Pixel centers are in origin and size

        # Apply
        self.geometry = geometry
