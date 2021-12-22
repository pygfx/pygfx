import numpy as np

from ._base import Geometry


def trimesh_geometry(mesh):
    """Create geometry from a trimesh Mesh."""
    kwargs = dict(
        positions=np.ascontiguousarray(mesh.vertices, dtype="f4"),
        indices=np.ascontiguousarray(mesh.faces, dtype="i4"),
        normals=np.ascontiguousarray(mesh.vertex_normals, dtype="f4"),
    )
    if mesh.visual.kind == "texture":
        kwargs["texcoords"] = np.ascontiguousarray(mesh.visual.uv, dtype="f4")
    elif mesh.visual.kind == "vertex":
        kwargs["colors"] = np.ascontiguousarray(mesh.visual.vertex_colors, dtype="f4")
    return Geometry(**kwargs)
