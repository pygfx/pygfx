"""
Utilities to work with gltf files
"""

import numpy as np
import pygfx as gfx


def load_gltf(path):
    """Load standard meshes from a gltf file.
    This function requires the trimesh library. Might not be complete yet.

    Parameters:
        path (str): the path to a gltf file.
    """

    import trimesh  # noqa

    scene = trimesh.load(path)
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        current = scene.geometry[geometry_name]
        current.apply_transform(transform)

    meshes = list(scene.geometry.values())
    return [_parse_mesh(m) for m in meshes]


def _parse_mesh(mesh):
    visual = mesh.visual
    visual.uv = visual.uv * np.array([1, -1]) + np.array([0, 1])  # uv.y = 1 - uv.y
    return gfx.Mesh(
        _parse_geometry(mesh),
        _parse_material(visual.material),
    )


def _parse_geometry(mesh):
    return gfx.trimesh_geometry(mesh)


def _parse_material(pbrmaterial):
    material = gfx.MeshStandardMaterial()
    material.map = _parse_texture(pbrmaterial.baseColorTexture)

    material.emissive = gfx.Color(*pbrmaterial.emissiveFactor)
    material.emissive_map = _parse_texture(pbrmaterial.emissiveTexture)

    metallic_roughness_map = _parse_texture(pbrmaterial.metallicRoughnessTexture)
    material.roughness = pbrmaterial.roughnessFactor or 1.0
    material.metalness = pbrmaterial.metallicFactor or 1.0
    material.roughness_map = metallic_roughness_map
    material.metalness_map = metallic_roughness_map

    material.normal_map = _parse_texture(pbrmaterial.normalTexture)
    material.normal_scale = (1, -1)

    material.ao_map = _parse_texture(pbrmaterial.occlusionTexture)
    material.ao_map_intensity = 1.0

    material.side = "FRONT"
    return material


def _parse_texture(pil_image):
    if pil_image is None:
        return None
    m = memoryview(pil_image.tobytes())
    m = m.cast(m.format, shape=(pil_image.size[0], pil_image.size[1], 3))
    tex = gfx.Texture(m, dim=2)
    return tex.get_view(address_mode="repeat", filter="linear")
