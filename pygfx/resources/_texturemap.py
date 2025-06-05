import numpy as np
from ..utils.trackable import Trackable
from ..resources import Buffer
from ..utils import array_from_shadertype


class TextureMap(Trackable):
    """A texture map.

    TextureMap is used to define how a texture should be used in a material.
    including which uv channel to use, filtering, and wrapping

    Parameters
    ----------
    texture : Texture
        The texture to use for this map.
    uv_channel : int, optional
        The uv channel of the texture to use.
        e.g. with `channel=7`, it will use the `geometry.texcoords7`.
        Default is 0, which uses the `geometry.texcoords`.
    filter : str, optional
        The filter to use for magnification, minification znd mipmap, if not being set separately.
        Can be "nearest" or "linear". Default is "linear".
    mag_filter : str, optional
        The magnification filter to use. Defaults to the value of ``filter``.
    min_filter : str, optional
        The minification filter to use. Defaults to the value of ``filter``.
    mipmap_filter : str, optional
        The mipmap filter to use. Defaults to the value of ``filter``.
    wrap : str, optional
        The wrap mode for both the s and t coordinates if not being set separately.
        Can be "clamp", "repeat", "mirror". Default is "repeat".
    wrap_s : str, optional
        The wrap mode for the s coordinate. Defaults to the value of ``wrap``.
    wrap_t : str, optional
        The wrap mode for the t coordinate. Defaults to the value of ``wrap``.
    """

    uniform_type = dict(
        transform="3x3xf4",
    )

    def __init__(
        self,
        texture,
        *,
        uv_channel=0,
        filter="linear",
        mag_filter=None,
        min_filter=None,
        mipmap_filter=None,
        wrap="repeat",
        wrap_s=None,
        wrap_t=None,
    ):
        super().__init__()
        self.texture = texture
        self.uv_channel = uv_channel
        self.mag_filter = mag_filter or filter
        self.min_filter = min_filter or filter
        self.mipmap_filter = mipmap_filter or filter
        self.wrap_s = wrap_s or wrap
        self.wrap_t = wrap_t or wrap

        self._offset = (0, 0)
        self._scale = (1, 1)
        self._rotation = 0

        self._store.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), force_contiguous=True
        )
        self.update_matrix()

    @property
    def texture(self):
        """The texture to use for this map."""
        return self._store.texture

    @texture.setter
    def texture(self, value):
        self._store.texture = value

    @property
    def uv_channel(self):
        """The channel of the texture to use."""
        return self._store.uv_channel

    @uv_channel.setter
    def uv_channel(self, value):
        self._store.uv_channel = value

    @property
    def mag_filter(self):
        """The magnification filter to use."""
        return self._store.mag_filter

    @mag_filter.setter
    def mag_filter(self, value):
        self._store.mag_filter = value

    @property
    def min_filter(self):
        """The minification filter to use."""
        return self._store.min_filter

    @min_filter.setter
    def min_filter(self, value):
        self._store.min_filter = value

    @property
    def mipmap_filter(self):
        """The mipmap filter to use."""
        return self._store.mipmap_filter

    @mipmap_filter.setter
    def mipmap_filter(self, value):
        self._store.mipmap_filter = value

    @property
    def wrap_s(self):
        """The wrap mode for the s coordinate."""
        return self._store.wrap_s

    @wrap_s.setter
    def wrap_s(self, value):
        self._store.wrap_s = value

    @property
    def wrap_t(self):
        """The wrap mode for the t coordinate."""
        return self._store.wrap_t

    @wrap_t.setter
    def wrap_t(self, value):
        self._store.wrap_t = value

    @property
    def offset(self):
        """The uv offset of the texture map.

        Note: You should call `update_matrix` after changing this value to update the matrix.
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value

    @property
    def scale(self):
        """The uv scale of the texture map.

        Note: You should call `update_matrix` after changing this value to update the matrix.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    @property
    def rotation(self):
        """The rotation of the texture map in uv plane.

        Note: You should call `update_matrix` after changing this value to update the matrix.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value

    def update_matrix_components(self, offset=None, scale=None, rotation=None):
        """Update the texture map's matrix with the given offset, scale and rotation.

        Parameters
        ----------
        offset : tuple of float, optional
            The offset of the texture map.
        scale : tuple of float, optional
            The scale of the texture map.
        rotation : float, optional
            The rotation of the texture map.
        """

        if offset is not None:
            self._offset = offset

        if scale is not None:
            self._scale = scale

        if rotation is not None:
            self._rotation = rotation

        self.update_matrix()

    # Currently, we let users to manage the matrix update timing themselves.
    # todo: auto update matrix when offset, scale, rotation are changed before the next draw call.
    def update_matrix(self):
        """Update the matrix of the texture map."""

        cos_r = np.cos(self._rotation)
        sin_r = np.sin(self._rotation)

        matrix = np.array(
            [
                [cos_r * self._scale[0], sin_r * self._scale[0], self._offset[0]],
                [-sin_r * self._scale[1], cos_r * self._scale[1], self._offset[1]],
                [0, 0, 1],
            ]
        )

        self.uniform_buffer.data["transform"][:, :3] = matrix.T
        self.uniform_buffer.update_full()

    @property
    def uniform_buffer(self):
        """The uniform buffer of the texture map."""
        return self._store.uniform_buffer
