"""
Utilities to load scenes from files, using trimesh.
"""

import pygfx as gfx


def load_scene(path):
    """Load a scene from a file.

    This function requires the trimesh library. Might not be complete yet.

    Parameters
    ----------
    path : str
        The location where the scene description is stored.

    Returns
    -------
    meshes : list
        A list of loaded meshes.

    """

    import trimesh  # noqa

    scene = trimesh.load(path)
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
