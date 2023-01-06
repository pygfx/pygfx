from ._mesh import MeshStandardMaterial
from ..resources import Texture
from ..utils import Color


def pillow_image(image, dim=2, **kwargs):
    """Create a Texture from a PIL.Image."""
    from PIL.Image import Image  # noqa

    if not isinstance(image, Image):
        raise NotImplementedError()

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = image.size + (im_channels,)

    m = m.cast(m.format, shape=buffer_shape)
    return Texture(m, dim=dim, **kwargs)


def trimesh_material(material):
    """Create Material from a trimesh Material."""
    from trimesh.visual.material import PBRMaterial  # noqa

    if not isinstance(material, PBRMaterial):
        raise NotImplementedError()

    gfx_material = MeshStandardMaterial()

    if material.baseColorTexture is not None:
        gfx_material.map = pillow_image(material.baseColorTexture).get_view(
            address_mode="repeat", filter="linear"
        )

    if material.emissiveFactor is not None:
        gfx_material.emissive = Color(*material.emissiveFactor)

    if material.emissiveTexture is not None:
        gfx_material.emissive_map = pillow_image(material.emissiveTexture).get_view(
            address_mode="repeat", filter="linear"
        )

    if material.metallicRoughnessTexture is not None:
        metallic_roughness_map = pillow_image(
            material.metallicRoughnessTexture
        ).get_view(address_mode="repeat", filter="linear")
        gfx_material.roughness_map = metallic_roughness_map
        gfx_material.metalness_map = metallic_roughness_map
        gfx_material.roughness = 1.0
        gfx_material.metalness = 1.0

    if material.roughnessFactor is not None:
        gfx_material.roughness = material.roughnessFactor

    if material.metallicFactor is not None:
        gfx_material.metalness = material.metallicFactor

    if material.normalTexture is not None:
        gfx_material.normal_map = pillow_image(material.normalTexture).get_view(
            address_mode="repeat", filter="linear"
        )
        # See: https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
        gfx_material.normal_scale = (1.0, -1.0)

    if material.occlusionTexture is not None:
        gfx_material.ao_map = pillow_image(material.occlusionTexture).get_view(
            address_mode="repeat", filter="linear"
        )

    gfx_material.side = "FRONT"
    return gfx_material
