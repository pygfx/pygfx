from ..utils.trackable import Trackable


class TextureMap(Trackable):
    """A texture map.

    TextureMap is used to define how a texture should be used in a material.
    including which uv channel to use, filtering, and wrapping

    Parameters
    ----------
    texture : Texture
        The texture to use for this map.
    channel : int, optional
        The uv channel of the texture to use. Default is 0.
    mag_filter : str, optional
        The magnification filter to use. Default is "linear".
    min_filter : str, optional
        The minification filter to use. Default is "linear".
    mipmap_filter : str, optional
        The mipmap filter to use. Default is "linear".
    wrap_s : str, optional
        The wrap mode for the s coordinate. Default is "repeat".
    wrap_t : str, optional
        The wrap mode for the t coordinate. Default is "repeat".
    """

    def __init__(
        self,
        texture,
        *,
        channel=0,
        mag_filter="linear",
        min_filter="linear",
        mipmap_filter="linear",
        wrap_s="repeat",
        wrap_t="repeat",
    ):
        super().__init__()
        self.texture = texture
        self.channel = channel
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.mipmap_filter = mipmap_filter
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t

    @property
    def texture(self):
        """The texture to use for this map."""
        return self._store.texture

    @texture.setter
    def texture(self, value):
        self._store.texture = value

    @property
    def channel(self):
        """The channel of the texture to use."""
        return self._store.channel

    @channel.setter
    def channel(self, value):
        self._store.channel = value

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