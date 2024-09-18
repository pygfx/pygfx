from ._mesh import MeshStandardMaterial, MeshPhongMaterial
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

    # If this is a palette image, convert it to RGBA
    if getattr(image, "mode", None) == "P":
        image = image.convert("RGBA")

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = image.size + (im_channels,)

    m = m.cast(m.format, shape=buffer_shape)
    return Texture(m, dim=dim, **kwargs)


def material_from_trimesh(x):
    """Convert a Trimesh object into a pygfx material.

    Parameters
    ----------
    x : trimesh.Material | trimesh.Visuals | trimesh.Trimesh
        Either the actual material to convert or an object containing
        the material.

    Returns
    -------
    converted : Material
        The converted material.

    """
    import trimesh  # noqa
    from trimesh.visual.material import PBRMaterial, SimpleMaterial  # noqa
    from trimesh.visual import ColorVisuals  # noqa

    # If this is a trimesh object, extract the visual
    if isinstance(x, trimesh.Trimesh):
        x = x.visual

    # If this is a visual, check if it has a material; this is the case
    # for e.g. TextureVisuals which should contain a SimpleMaterial.
    # ColorVisuals, on the other hand, do not have a material and are
    # typically used to store vertex or face colors - we will deal with
    # them below
    if isinstance(x, trimesh.visual.base.Visuals) and hasattr(x, "material"):
        x = x.material

    if isinstance(x, PBRMaterial):
        material = x
        gfx_material = MeshStandardMaterial()

        if material.baseColorTexture is not None:
            gfx_material.map = texture_from_pillow_image(material.baseColorTexture)

        if material.emissiveFactor is not None:
            gfx_material.emissive = Color(*material.emissiveFactor)

        if material.emissiveTexture is not None:
            gfx_material.emissive_map = texture_from_pillow_image(
                material.emissiveTexture
            )

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
            # See: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
            gfx_material.normal_scale = (1.0, -1.0)

        if material.occlusionTexture is not None:
            gfx_material.ao_map = texture_from_pillow_image(material.occlusionTexture)

        gfx_material.side = "front"
    elif isinstance(x, SimpleMaterial):
        material = x
        gfx_material = MeshPhongMaterial(color=material.ambient / 255)

        gfx_material.shininess = material.glossiness
        gfx_material.specular = Color(*(material.specular / 255))

        # Note: `material.image` can exist but be None
        if getattr(material, "image", None):
            gfx_material.map = texture_from_pillow_image(material.image)

        gfx_material.side = "front"
    elif isinstance(x, ColorVisuals):
        # ColorVisuals are typically used for vertex or face colors but they can be
        # also be undefined in which case it's just a default material
        gfx_material = MeshPhongMaterial(color=x.main_color / 255)

        gfx_material.shininess = x.defaults["material_shine"]
        gfx_material.specular = Color(*(x.defaults["material_specular"] / 255))

        gfx_material.side = "front"

        if x.kind == "vertex":
            gfx_material.color_mode = "vertex"
        elif x.kind == "face":
            gfx_material.color_mode = "face"
    else:
        raise NotImplementedError(f"Conversion of {type(x)} is not supported.")

    return gfx_material
