from ..utils.trackable import Trackable


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
