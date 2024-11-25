from importlib.util import find_spec

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
    from trimesh import Trimesh

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


def geometry_from_open3d(x) -> Geometry:
    """
    Convert an Open3D geometry object to a pygfx Geometry.

    This function handles the conversion of Open3D geometry objects such as
    `open3d.geometry.TriangleMesh <https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html>`_
    or `open3d.geometry.PointCloud <https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html>`_
    into pygfx Geometry objects. Depending on the input type, it delegates the conversion
    to specialized conversion functions.

    Parameters
    ----------
    x : open3d.geometry.TriangleMesh or open3d.geometry.PointCloud
        The geometry to be converted into a pygfx Geometry.

    Returns
    -------
    converted_geometry : Geometry
        A Geometry object representing the given Open3D geometry.

    Raises
    ------
    ImportError
        If the open3d library is not installed.
    NotImplementedError
        If the input geometry type is not supported.
    """
    if not find_spec("open3d"):
        raise ImportError(
            "The `open3d` library is required for this function: pip install open3d"
        )

    from open3d import geometry as o3d_geometry

    if isinstance(x, o3d_geometry.TriangleMesh):
        return geometry_from_open3d_triangle_mesh(x)
    elif isinstance(x, o3d_geometry.PointCloud):
        return geometry_from_open3d_point_cloud(x)
    else:
        raise NotImplementedError(
            "Conversion for the provided Open3D geometry type is not implemented."
        )


def geometry_from_open3d_triangle_mesh(x) -> Geometry:
    """
    Convert an Open3D TriangleMesh object to a pygfx Geometry.

    This function creates a pygfx Geometry object from the given
    `open3d.geometry.TriangleMesh <https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html>`_
    by extracting vertices, triangles, normals, and texture coordinates (if available).

    Parameters
    ----------
    x : open3d.geometry.TriangleMesh
        The TriangleMesh object to be converted into a pygfx Geometry.

    Returns
    -------
    converted_mesh : Geometry
        A Geometry object representing the given TriangleMesh.

    Raises
    ------
    ImportError
        If the open3d library is not installed.
    NotImplementedError
        If the input is not an instance of open3d.geometry.TriangleMesh.
    """
    if not find_spec("open3d"):
        raise ImportError(
            "The `open3d` library is required for this function: pip install open3d"
        )

    from open3d import geometry as o3d_geometry

    if not isinstance(x, o3d_geometry.TriangleMesh):
        raise NotImplementedError(
            "Input must be an instance of open3d.geometry.TriangleMesh."
        )

    vertices = np.ascontiguousarray(x.vertices, dtype=np.float32)
    triangles = np.ascontiguousarray(x.triangles, dtype="i4")

    kwargs = dict(positions=vertices, indices=triangles)

    # Add normals if available
    if len(x.vertex_normals) > 0:
        kwargs["normals"] = np.ascontiguousarray(x.vertex_normals, dtype=np.float32)

    # Add UV coordinates if available
    if len(x.triangle_uvs) > 0:
        triangle_uvs = np.ascontiguousarray(x.triangle_uvs, dtype=np.float32)

        vertex_uvs = np.zeros((len(vertices), 2), np.float32)
        vertex_uvs[triangles.flat] = triangle_uvs

        # Adjust UVs for rendering systems
        vertex_uvs_wgpu = (vertex_uvs * np.array([1, -1]) + np.array([0, 1])).astype(
            np.float32
        )
        kwargs["texcoords"] = vertex_uvs_wgpu

    return Geometry(**kwargs)


def geometry_from_open3d_point_cloud(x) -> Geometry:
    """
    Convert an Open3D PointCloud object to a pygfx Geometry.

    This function creates a pygfx Geometry object from the given
    `open3d.geometry.PointCloud <https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html>`_
    by extracting points, colors, and normals (if available).

    Parameters
    ----------
    x : open3d.geometry.PointCloud
        The PointCloud object to be converted into a pygfx Geometry.

    Returns
    -------
    converted_geometry : Geometry
        A Geometry object representing the given PointCloud.

    Raises
    ------
    ImportError
        If the open3d library is not installed.
    NotImplementedError
        If the input is not an instance of open3d.geometry.PointCloud.
    """
    if not find_spec("open3d"):
        raise ImportError(
            "The `open3d` library is required for this function: pip install open3d"
        )

    from open3d import geometry as o3d_geometry

    if not isinstance(x, o3d_geometry.PointCloud):
        raise NotImplementedError(
            "Input must be an instance of open3d.geometry.PointCloud."
        )

    points = np.ascontiguousarray(x.points, dtype=np.float32)
    kwargs = dict(positions=points)

    # Add colors if available
    if len(x.colors) > 0:
        kwargs["colors"] = np.ascontiguousarray(x.colors, dtype=np.float32)

    # Add normals if available
    if len(x.normals) > 0:
        kwargs["normals"] = np.ascontiguousarray(x.normals, dtype=np.float32)

    return Geometry(**kwargs)
