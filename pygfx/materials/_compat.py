from importlib.util import find_spec

import numpy as np

from ._mesh import MeshStandardMaterial, MeshPhongMaterial
from ..resources import Texture
from ..utils.color import Color


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

    from PIL.Image import Image

    if not isinstance(image, Image):
        raise NotImplementedError()

    # If this is a palette image, convert it to RGBA
    if getattr(image, "mode", None) == "P":
        image = image.convert("RGBA")

    m = memoryview(image.tobytes())

    im_channels = len(image.getbands())
    buffer_shape = (*image.size, im_channels)

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
    import trimesh
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    from trimesh.visual import ColorVisuals

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

        default_mat = getattr(trimesh.visual.color, "DEFAULT_MAT", None)
        if default_mat is None:
            default_mat = x.defaults  # trimesh < 4.6.3
        gfx_material.shininess = default_mat["material_shine"]
        gfx_material.specular = Color(*(default_mat["material_specular"] / 255))

        gfx_material.side = "front"

        if x.kind == "vertex":
            gfx_material.color_mode = "vertex"
        elif x.kind == "face":
            gfx_material.color_mode = "face"
    else:
        raise NotImplementedError(f"Conversion of {type(x)} is not supported.")

    return gfx_material


def material_from_open3d(x):
    """Convert an Open3D MaterialRecord object into a pygfx material.

    Parameters
    ----------
    x : open3d.visualization.rendering.MaterialRecord | open3d.geometry.Geometry3D
        Either the actual Open3D MaterialRecord or an object containing
        a material (e.g., geometry).

    Returns
    -------
    converted : Material
        The converted pygfx material.

    """
    if not find_spec("open3d"):
        raise ImportError(
            "The `open3d` library is required for this function: pip install open3d"
        )

    import open3d as o3d

    # Ensure the input is a MaterialRecord
    if not isinstance(x, o3d.visualization.rendering.MaterialRecord):
        raise NotImplementedError("Input must be an Open3D MaterialRecord")

    # Determine which pygfx material to create based on shader type
    if x.shader in ["defaultLit", "litPBR"]:
        gfx_material = MeshStandardMaterial()

        # Set base metallic and roughness values
        gfx_material.metalness = getattr(x, "base_metallic", 1.0)
        gfx_material.roughness = getattr(x, "base_roughness", 1.0)

        # Set albedo color if available
        if x.base_color is not None:
            albedo_color = (
                np.array(x.base_color[:3]) / 255
            )  # Convert to normalized values
            gfx_material.color = Color(*albedo_color)

        # Handle textures if available
        if x.albedo_img is not None:
            gfx_material.map = Texture(np.ascontiguousarray(x.albedo_img), dim=2)

        if x.normal_img is not None:
            gfx_material.normal_map = Texture(np.ascontiguousarray(x.normal_img), dim=2)
            gfx_material.normal_scale = (1.0, -1.0)

        if x.ao_img is not None:
            gfx_material.ao_map = Texture(np.ascontiguousarray(x.ao_img), dim=2)

        if x.metallic_img is not None:
            gfx_material.metalness_map = Texture(
                np.ascontiguousarray(x.metallic_img), dim=2
            )

        if x.roughness_img is not None:
            gfx_material.roughness_map = Texture(
                np.ascontiguousarray(x.roughness_img), dim=2
            )

        gfx_material.side = "front"

    elif x.shader == "unlit":
        gfx_material = MeshPhongMaterial()
        if x.base_color is not None:
            base_color = (
                np.array(x.base_color[:3]) / 255
            )  # Convert to normalized values
            gfx_material.color = Color(*base_color)

        gfx_material.shininess = getattr(x, "base_reflectance", 0.5)
        gfx_material.specular = Color(1, 1, 1)  # Default specular color

        if x.albedo_img is not None:
            gfx_material.map = Texture(np.ascontiguousarray(x.albedo_img), dim=2)

        gfx_material.side = "front"

    else:
        raise NotImplementedError(f"Shader type {x.shader} is not supported.")

    return gfx_material
