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

    Returns
    -------
    meshes : list
        A list of loaded meshes.

    """

    import trimesh  # noqa

    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Trimesh):
        m = gfx.Mesh(
            gfx.geometry_from_trimesh(scene),
            gfx.MeshPhongMaterial(),
        )
        return [m]

    elif isinstance(scene, trimesh.Scene):
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            current = scene.geometry[geometry_name]
            current.apply_transform(transform)

        return [
            gfx.Mesh(
                gfx.geometry_from_trimesh(m),
                gfx.material_from_trimesh(m.visual.material),
            )
            for m in scene.geometry.values()
        ]
    else:
        raise ValueError(f"Unexpected trimesh data: {scene.__class__.__name__}")
