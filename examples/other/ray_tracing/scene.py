import numpy as np
import pygfx as gfx


class Sphere(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.zeros(
            1,
            dtype=np.dtype(
                [
                    ("center", np.float32, 3),
                    ("radius", np.float32),
                    ("material_index", np.uint32),
                    ("__padding", np.uint32, 3),
                ]
            ),
        ).view(cls)

    def __init__(self, center=(0, 0, 0), radius=1, material_index=0):
        super().__init__()
        self["center"] = center
        self["radius"] = radius
        self["material_index"] = material_index


class Triangle(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.zeros(
            1,
            dtype=np.dtype(
                [
                    ("p0", np.float32, 3),
                    ("__padding_0", np.uint32, 1),
                    ("p1", np.float32, 3),
                    ("__padding_1", np.uint32, 1),
                    ("p2", np.float32, 3),
                    ("__padding_2", np.uint32, 1),
                    ("n0", np.float32, 3),
                    ("__padding_3", np.uint32, 1),
                    ("n1", np.float32, 3),
                    ("__padding_4", np.uint32, 1),
                    ("n2", np.float32, 3),
                    ("material_index", np.uint32),
                ]
            ),
        ).view(cls)

    def __init__(
        self,
        v0=(0, 0, 0),
        v1=(0, 0, 0),
        v2=(0, 0, 0),
        n0=(0, 0, 0),
        n1=(0, 0, 0),
        n2=(0, 0, 0),
        material_index=0,
    ):
        super().__init__()
        self["p0"] = v0
        self["p1"] = v1
        self["p2"] = v2
        self["n0"] = n0
        self["n1"] = n1
        self["n2"] = n2
        self["material_index"] = material_index


class Material(np.ndarray, gfx.Material):
    def __new__(cls, *args, **kwargs):
        return np.zeros(
            1,
            dtype=np.dtype(
                [
                    ("color", np.float32, 3),
                    ("metallic", np.float32),
                    ("emissive", np.float32, 3),
                    ("roughness", np.float32),
                    ("ior", np.float32),
                    ("__padding", np.uint32, 3),
                ]
            ),
        ).view(cls)

    def __init__(
        self, color=(1, 1, 1), metallic=0, emissive=(0, 0, 0), roughness=1.0, ior=0.0
    ):
        super().__init__()
        self["color"] = color
        self["metallic"] = metallic
        self["emissive"] = emissive
        self["roughness"] = roughness
        self["ior"] = ior


def parse_gfx_scene(s: gfx.Scene):
    materials = []
    triangles = []
    materials_cache = []

    def parse_node(node):
        if isinstance(node, gfx.Mesh):
            world_matrix = node.world.matrix
            world_matrix_inv = node.world.inverse_matrix
            normal_matrix = world_matrix_inv[:3, :3].T

            geometry = node.geometry
            vertices = geometry.positions.data
            normals = geometry.normals.data

            vertices = np.hstack(
                (vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype))
            )
            vertices = (world_matrix @ vertices.T).T[:, :3]

            normals = (normal_matrix @ normals.T).T
            normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

            indices = geometry.indices.data
            material = node.material

            if material is not None:
                if material not in materials_cache:
                    materials_cache.append(material)

                material_index = materials_cache.index(material)

            for i in range(len(indices)):
                v0 = vertices[indices[i][0]]
                v1 = vertices[indices[i][1]]
                v2 = vertices[indices[i][2]]

                n0 = normals[indices[i][0]]
                n1 = normals[indices[i][1]]
                n2 = normals[indices[i][2]]

                triangle = Triangle(v0, v1, v2, n0, n1, n2, material_index)

                triangles.append(triangle)

    s.traverse(parse_node)

    for i in range(len(materials_cache)):
        material = materials_cache[i]

        if isinstance(material, Material):
            materials.append(material)
        else:
            color = material.color
            if isinstance(material, gfx.MeshPhongMaterial):
                metallic = 1.0
                roughness = 0.0
                emissive = material.emissive
            elif isinstance(material, gfx.MeshStandardMaterial):
                metallic = material.metalness
                roughness = material.roughness
                emissive = material.emissive

            materials.append(Material(color, metallic, emissive, roughness))

    print(f"triangles: {len(triangles)}, materials: {len(materials)}")

    # concatenate arrays
    triangles_b = np.concatenate(triangles)
    materials_b = np.concatenate(materials)

    return triangles_b, materials_b
