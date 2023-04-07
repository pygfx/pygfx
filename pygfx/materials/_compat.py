from ._mesh import MeshStandardMaterial
from ..utils.color import Color
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


def material_from_trimesh(material):
    """Convert a Trimesh material object into a pygfx material.

    Parameters
    ----------
    material : trimesh.Material
        The material to convert.

    Returns
    -------
    converted : Material
        The converted material.

    """
    from trimesh.visual.material import PBRMaterial  # noqa

    if not isinstance(material, PBRMaterial):
        raise NotImplementedError()

    gfx_material = MeshStandardMaterial()

    if material.baseColorTexture is not None:
        gfx_material.map = texture_from_pillow_image(material.baseColorTexture)

    if material.emissiveFactor is not None:
        gfx_material.emissive = Color(*material.emissiveFactor)

    if material.emissiveTexture is not None:
        gfx_material.emissive_map = texture_from_pillow_image(material.emissiveTexture)

    if material.metallicRoughnessTexture is not None:
        metallic_roughness_map = texture_from_pillow_image(
            material.metallicRoughnessTexture
        )
        gfx_material.roughness_map = metallic_roughness_map
        gfx_material.metalness_map = metallic_roughness_map
        gfx_material.roughness = 1.0
        gfx_material.metalness = 1.0

    if material.roughnessFactor is not None:
        gfx_material.roughness = material.roughnessFactor

    if material.metallicFactor is not None:
        gfx_material.metalness = material.metallicFactor

    if material.normalTexture is not None:
        gfx_material.normal_map = texture_from_pillow_image(material.normalTexture)
        # See: https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
        gfx_material.normal_scale = (1.0, -1.0)

    if material.occlusionTexture is not None:
        gfx_material.ao_map = texture_from_pillow_image(material.occlusionTexture)

    gfx_material.side = "FRONT"
    return gfx_material
