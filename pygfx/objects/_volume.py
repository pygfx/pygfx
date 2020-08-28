from ._base import WorldObject
from ..geometries import BoxGeometry

from ..resources import Texture, TextureView


class Volume(WorldObject):
    """A volume represents a 3D image in space. It has an implicit geometry
    based on the shape of the texture.
    """

    def __init__(self, texture, material):
        super().__init__()

        self.material = material
        self.texture = texture

        # Create geometry
        self.geometry = BoxGeometry(*self.texture.size)
        # self.geometry.texcoords = ...

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
