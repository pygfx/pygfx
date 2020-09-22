from ._base import WorldObject
from ..geometries import BoxGeometry
from ..resources import Buffer, Texture, TextureView


class Volume(WorldObject):
    """A volume represents a 3D image in space. It has an implicit geometry
    based on the shape of the texture.
    """

    def __init__(self, texture, material):
        super().__init__()

        self.material = material
        self.texture = texture

        # Create box geometry, and map to 0..1
        geometry = BoxGeometry(1, 1, 1)
        geometry.positions.data[:, :3] += 0.5
        # This is our 3D texture coords
        geometry.texcoords = Buffer(
            geometry.positions.data[:, :3].copy(), usage="vertex|storage"
        )
        # Map to volume size
        for i in range(3):
            geometry.positions.data[:, i] *= self.texture.size[i]
        self.geometry = geometry

        # todo: how to handle spacing and origin, do we express these using transorms, or directly?

    @property
    def texture(self):
        """The 3D texture (or texture view) representing the volume."""
        return self.material.map.texture

    @texture.setter
    def texture(self, texture):
        if isinstance(texture, TextureView):
            self.material.map = texture
        elif isinstance(texture, Texture):
            self.material.map = TextureView(
                texture, address_mode="clamp", filter="linear"
            )
        else:
            raise TypeError("Volume texture must be a Texture or TextureView.")
