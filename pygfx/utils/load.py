"""
Utilities to load scenes from files, using trimesh.
"""

import pygfx as gfx


def load_scene(path):
    raise DeprecationWarning(
        "The load_scene() function is replaced with load_meshes()."
    )


def load_mesh(path):
    """Load a mesh from a file.

    Parameters
    ----------
    path : str
        The location where the mesh is stored.

    Returns
    -------
    mesh : Mesh
        A pygfx Mesh object. If the file does not contain exactly one mesh, an error is raised.
    """
    meshes = load_meshes(path)
    if len(meshes) != 1:
        raise ValueError(
            f"Found {len(meshes)} meshes instead of 1 in '{path}'. Use `load_meshes()` instead. "
        )
    return meshes[0]


def load_meshes(path):
    """Load meshes from a file.

    This function requires the trimesh library.

    Parameters
    ----------
    path : str
        The location where the mesh or scene is stored.
        Can be a local file path or a URL.

    Returns
    -------
    meshes : list
        A list of loaded meshes.

    """
    import trimesh  # noqa

    # Trimesh's load() performs a similar check and refers
    # loading from URLs to load_remote()
    if "https://" in str(path) or "http://" in str(path):
        scene = trimesh.load_remote(path)
    else:
        scene = trimesh.load(path)

    # If this is a single object we can just convert and return it
    if isinstance(scene, trimesh.Trimesh):
        m = gfx.Mesh(
            gfx.geometry_from_trimesh(scene),
            gfx.MeshPhongMaterial(),
        )
        return [m]
    # If this is a scene, we need to properly parse it
    elif isinstance(scene, trimesh.Scene):
        # Scene consists of geometries (which in trimesh contain both
        # the vertices/faces as well as the mateiral) and a scene graph.
        # Each node in the graph consists of a geometry and a transform
        # A geometry can be used by more than one node in the graph.
        # Here we will convert each node in the graph to a pygfx.Mesh
        # but in the future it might be better to use pygfx.InstancedMesh.
        gfx_geometries = {}
        gfx_materials = {}
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]

            # Convert each geometry only once
            if geometry_name not in gfx_geometries:
                m = scene.geometry[geometry_name]
                gfx_geometries[geometry_name] = gfx.geometry_from_trimesh(m)
                if hasattr(m.visual, "material"):
                    gfx_materials[geometry_name] = gfx.material_from_trimesh(
                        m.visual.material
                    )
                else:
                    gfx_materials[geometry_name] = gfx.MeshStandardMaterial()

            # Generate a visual for each node
            mesh = gfx.Mesh(gfx_geometries[geometry_name], gfx_materials[geometry_name])
            mesh.local.matrix = transform  # set the node's transform

            meshes.append(mesh)
        return meshes
    else:
        raise ValueError(f"Unexpected trimesh data: {scene.__class__.__name__}")
