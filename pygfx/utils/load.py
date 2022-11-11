"""
Utilities to load scenes from files, using trimesh.
"""

import pygfx as gfx


def load_scene(path):
    """Load a scene from a file.
    This function requires the trimesh library. Might not be complete yet.

    Parameters:
        path (str): the path to a file.

    Returns:
        meshes (list): list of meshes
    """

    import trimesh  # noqa

    scene = trimesh.load(path)
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        current = scene.geometry[geometry_name]
        current.apply_transform(transform)

    return [
        gfx.Mesh(
            gfx.trimesh_geometry(m),
            gfx.trimesh_material(m.visual.material),
        )
        for m in scene.geometry.values()
    ]
