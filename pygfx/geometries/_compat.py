import numpy as np

from ._base import Geometry


def geometry_from_trimesh(mesh):
    """Convert a Trimesh geometry object to pygfx geometry.

    Creates a Geometry object from the given `trimesh.Trimesh
    <https://trimsh.org/trimesh.html#trimesh.Trimesh>`_ object.

    Parameters
    ----------
    mesh : Trimesh
        The mesh to be converted into a geometry.

    Returns
    -------
    converted_mesh : Geometry
        A Geometry object representing the given mesh.

    """
    from trimesh import Trimesh  # noqa

    if not isinstance(mesh, Trimesh):
        raise NotImplementedError()

    kwargs = dict(
        positions=np.ascontiguousarray(mesh.vertices, dtype="f4"),
        indices=np.ascontiguousarray(mesh.faces, dtype="i4"),
        normals=np.ascontiguousarray(mesh.vertex_normals, dtype="f4"),
    )
    # Note: some trimesh visuals have type texture but no UV coordinates
    if mesh.visual.kind == "texture" and getattr(mesh.visual, "uv", None) is not None:
        # convert the uv coordinates from opengl to wgpu conventions.
        # wgpu uses the D3D and Metal coordinate systems.
        # the coordinate origin is in the upper left corner, while the opengl coordinate
        # origin is in the lower left corner.
        # trimesh loads textures according to the opengl coordinate system.
        wgpu_uv = mesh.visual.uv * np.array([1, -1]) + np.array(
            [0, 1]
        )  # uv.y = 1 - uv.y
        kwargs["texcoords"] = np.ascontiguousarray(wgpu_uv, dtype="f4")
    elif mesh.visual.kind == "vertex":
        kwargs["colors"] = np.ascontiguousarray(mesh.visual.vertex_colors, dtype="f4")
    elif mesh.visual.kind == "face":
        kwargs["colors"] = np.ascontiguousarray(mesh.visual.face_colors, dtype="f4")

    # todo: support vertex attribute 'tangent'

    return Geometry(**kwargs)


def geometry_from_open3d(mesh):
    """Convert an Open3D geometry object to pygfx geometry.

    Creates a Geometry object from the given `open3d.geometry.TriangleMesh
    <https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html>`_ object.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        The mesh to be converted into a geometry.

    Returns
    -------
    converted_mesh : Geometry
        A Geometry object representing the given mesh.

    """

    from open3d import geometry as o3d_geometry  # noqa

    if not isinstance(mesh, o3d_geometry.TriangleMesh):
        raise NotImplementedError()

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(mesh.triangles, dtype="i4")

    kwargs = dict(positions=vertices, indices=triangles)

    # normals
    if len(mesh.vertex_normals) > 0:
        kwargs["normals"] = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)

    # uvs
    if len(mesh.triangle_uvs) > 0:
        triangle_uvs = np.ascontiguousarray(mesh.triangle_uvs, dtype=np.float32)

        vertex_uvs = np.zeros((len(vertices), 2), np.float32)
        vertex_uvs[triangles.flat] = triangle_uvs

        vertex_uvs_wgpu = (vertex_uvs * np.array([1, -1]) + np.array([0, 1])).astype(
            np.float32
        )  # uv.y = 1 - uv.y
        kwargs["texcoords"] = vertex_uvs_wgpu

    return Geometry(**kwargs)
