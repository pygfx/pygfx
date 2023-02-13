from ..resources import Texture


def texture_from_pillow_image(image, dim=2, **kwargs):
    """Pillow Image texture.

    Create a Texture from a PIL.Image.

    Parameters
    ----------
    image : Image
        The `PIL.Image
        <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_
        to convert into a texture.
    dim : int
        The number of spatial dimensions of the image.
    kwargs : Any
        Additional kwargs are forwarded to :class:`pygfx.Texture`.

    Returns
    -------
    image_texture : Texture
        A texture object representing the given image.

    """

    from PIL.Image import Image  # noqa

    if not isinstance(image, Image):
        raise NotImplementedError()

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = image.size + (im_channels,)

    m = m.cast(m.format, shape=buffer_shape)
    return Texture(m, dim=dim, **kwargs)
