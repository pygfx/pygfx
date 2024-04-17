"""
Utilities to load scenes from files, using trimesh.
"""

import pygfx as gfx
import numpy as np

from importlib.util import find_spec


def load_meshes(path, remote_ok=False):
    """Load meshes from a file.

    Deprecated: use load_mesh() instead.

    Parameters
    ----------
    path : str
        The location where the mesh is stored.
    remote_ok : bool
        Whether to allow loading meshes from URLs. Default is False.

    Returns
    -------
    meshes : list
        A list of loaded meshes.

    """
    raise DeprecationWarning("The load_meshes() function is replaced with load_mesh().")


def load_mesh(path, remote_ok=False):
    """Load mesh(es) from a file.

    This function requires the trimesh library.

    Parameters
    ----------
    path : str
        The location where the mesh or scene is stored.
        Can be a local file path or a URL.
    remote_ok : bool
        Whether to allow loading meshes from URLs. Default is False.

    Returns
    -------
    meshes : list
        A list of pygfx.Meshes.

    See Also
    --------
    load_scene
        Loads the entire scene (lights, graph, etc.) from a file.

    """
    if not find_spec("trimesh"):
        raise ImportError(
            "The `trimesh` library is required to load meshes: pip install trimesh"
        )

    import trimesh  # noqa

    # Trimesh's load() performs a similar check and refers
    # loading from URLs to load_remote()
    if "https://" in str(path) or "http://" in str(path):
        if not remote_ok:
            raise ValueError(
                "Loading meshes from URLs is disabled. "
                "Set remote_ok=True to allow loading from URLs."
            )
        if not find_spec("httpx"):
            raise ImportError(
                "The `httpx` library is required to load meshes from URLs: pip install httpx"
            )
        scene = trimesh.load_remote(path)
    else:
        scene = trimesh.load(path)

    return meshes_from_trimesh(scene, apply_transforms=True)


def load_scene(
    path,
    flatten=False,
    meshes=True,
    materials=True,
    volumes=True,
    lights="auto",
    camera="auto",
    remote_ok=False,
):
    """Load file into a scene.

    When reading file formats that can contain more than just meshes (e.g. `.glb`) this function will
    attempt to import the entire scene, including lights, cameras, volumes and materials. If the file
    format is mesh-only (e.g. `.stl`) or volume-only (e.g. `.binvox`) we will instead construct a
    minimal scene.

    This function requires the trimesh library.

    Parameters
    ----------
    path : str
        The filepath. Can be a local file or a URL.
    flatten : bool
        If True, will ignore any hierarchical structure in the scene graph and
        instead import as a flat list of objects. Default is False.
    meshes : bool
        Whether to load meshes. Default is True.
    materials : bool
        Whether to load materials. Default is True.
    volumes : bool
        Whether to load 3D image volumes. Default is True.
    lights :  "auto" | "file" | "none"
        Whether to import lights. "auto" (default) will always add lights to the scene
        even if the file does not contain any; "file" will import only lights defined in
        the file; "none" will not add any lights to the scene.
    camera :  "auto" | "file" | "none"
        Whether to import camera. "auto" (default) will always add a camera to the scene
        even if the file does not contain any; "file" will import only cameras defined in
        the file; "none" will not add any cameras to the scene.
    remote_ok : bool
        Whether to allow loading files from URLs. Default is False.

    Returns
    -------
    scene : pygfx.Scene
        The loaded scene.

    See Also
    --------
    load_mesh
        Returns the flat meshes contained in a file.

    """
    if not find_spec("trimesh"):
        raise ImportError(
            "The `trimesh` library is required to load scenes: pip install trimesh"
        )

    if lights not in ("auto", "file", "none"):
        raise ValueError(f"Invalid value for `lights`: {lights}")

    if camera not in ("auto", "file", "none"):
        raise ValueError(f"Invalid value for `camera`: {camera}")

    import trimesh  # noqa

    # Trimesh's load() performs a similar check and refers
    # loading from URLs to load_remote()
    if "https://" in str(path) or "http://" in str(path):
        if not remote_ok:
            raise ValueError(
                "Loading scenes from URLs is disabled. "
                "Set remote_ok=True to allow loading from URLs."
            )

        if not find_spec("httpx"):
            raise ImportError(
                "The `httpx` library is required to load meshes from URLs: pip install httpx"
            )
        tm_scene = trimesh.load_remote(path)
    else:
        tm_scene = trimesh.load(path)

    return scene_from_trimesh(
        tm_scene,
        flatten=flatten,
        meshes=meshes,
        volumes=volumes,
        materials=materials,
        lights=lights,
        camera=camera,
    )


def meshes_from_trimesh(scene, materials=True, apply_transforms=True):
    """Converts a trimesh scene into a flat list of pygfx Mesh objects.

    Parameters
    ----------
    scene : trimesh.Scene
        The scene to convert.
    materials : bool
        Whether to import materials. If False, a standard material will be created. Default is True.
    apply_transforms : bool
        Whether to apply the scene graph transforms directly to the meshes. Default is True.

    Returns
    -------
    meshes : list
        A list of loaded meshes.

    """
    import trimesh  # noqa

    # If this is a single object we can just convert and return it
    if isinstance(scene, trimesh.Trimesh):
        m = gfx.Mesh(
            gfx.geometry_from_trimesh(scene),
            gfx.material_from_trimesh(scene),
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

        # Extract the geometries and materials
        gfx_geometries, gfx_materials = objects_from_trimesh(scene, materials=materials)

        # Generate a visual for each node
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]

            mesh = gfx.Mesh(gfx_geometries[geometry_name], gfx_materials[geometry_name])

            if apply_transforms:
                mesh.local.matrix = transform  # set the node's transform

            meshes.append(mesh)
        return meshes
    else:
        raise ValueError(f"Unexpected trimesh data: {scene.__class__.__name__}")


def objects_from_trimesh(scene, materials=True, de_duplicate=True):
    """Extract geometries and materials from a trimesh scene.

    Parameters
    ----------
    scene : trimesh.Scene
        The scene to convert.
    materials : bool
        Whether to import materials. If False, a default material will be created.
    de_duplicate : bool
        Whether to deduplicate geometries and materials. This may mean that
        multiple objects in the scene will share the same geometry and/or material.

    Returns
    -------
    gfx_geometries : dict
        A dictionary of geometry objects. Keys are the names of the geometries.
    gfx_materials : dict
        A dictionary of material objects. Keys are the names of the geometries.

    """
    trimesh_to_gfx_geometries = {}  # cache for geometries
    trimesh_to_gfx_materials = {}  # cache for materials
    gfx_geometries = {}
    gfx_materials = {}
    for name, mesh in scene.geometry.items():
        # Convert mesh
        if de_duplicate and mesh in trimesh_to_gfx_materials:
            gfx_geometries[name] = trimesh_to_gfx_geometries[mesh]
        else:
            gfx_geometries[name] = trimesh_to_gfx_geometries[mesh] = (
                gfx.geometry_from_trimesh(mesh)
            )

        # If mesh has a material, convert it
        if hasattr(mesh.visual, "material") and materials:
            if de_duplicate and mesh.visual.material in trimesh_to_gfx_materials:
                gfx_materials[name] = trimesh_to_gfx_materials[mesh.visual.material]
            else:
                gfx_materials[name] = trimesh_to_gfx_materials[mesh.visual.material] = (
                    gfx.material_from_trimesh(mesh.visual.material)
                )
        # If not, use a default material
        else:
            gfx_materials[name] = gfx.MeshStandardMaterial()

    return gfx_geometries, gfx_materials


def scene_from_trimesh(
    tm_scene,
    flatten=False,
    meshes=True,
    materials=True,
    volumes=True,
    lights="auto",
    camera="auto",
    background=True,
):
    """Convert a trimesh scene into a pygfx scene.

    Parameters
    ----------
    tm_scene : trimesh.Scene
        A trimesh Scene.
    flatten : bool
        If True, will ignore any hierarchical structure in the scene graph and
        instead import as a flat list of objects. Default is False.
    meshes : bool
        Whether to import meshes. Default is True.
    materials : bool
        Whether to import materials. Default is True.
    volumes : bool
        Whether to load 3D image volumes. Default is True.
    lights :  "auto" | "file" | "none"
        Whether to import lights. "auto" (default) will always add lights to the scene
        even if the file does not contain any; "file" will import only lights defined in
        the file; "none" will not add any lights to the scene.
    camera :  "auto" | "file" | "none"
        Whether to import camera. "auto" (default) will always add a camera to the scene
        even if the file does not contain any; "file" will import only cameras defined in
        the file; "none" will not add any cameras to the scene.
    background : bool
        Trimesh scenes typically have a white background. If True, we will add a white background
        to the pygfx scene. If False, no background will be added. Default is True.

    Returns
    -------
    scene : pygfx.Scene
        The scene converted to pygfx.

    """
    if not find_spec("trimesh"):
        raise ImportError(
            "The `trimesh` library is required to load scenes: pip install trimesh"
        )

    if lights not in ("auto", "file", "none"):
        raise ValueError(f"Invalid value for `lights`: {lights}")

    if camera not in ("auto", "file", "none"):
        raise ValueError(f"Invalid value for `camera`: {camera}")

    import trimesh  # noqa

    # Basic scene setup
    gfx_scene = gfx.Scene()

    # Convet single meshes into a scene (this makes the code below much simpler)
    if isinstance(tm_scene, trimesh.Trimesh):
        tm_scene = trimesh.Scene(geometry=tm_scene)
    elif isinstance(tm_scene, trimesh.voxel.VoxelGrid):
        # VoxelGrids can not be in a trimesh.Scene directly (trimesh renders them
        # as collection of cube meshes), so we need to handle them separately
        if volumes:
            vol = _volume_from_voxelgrid(tm_scene)
            gfx_scene.add(vol)

        # Generate an empty scene so we can continue with the rest of the function
        # (i.e. potentially add lights and camera)
        tm_scene = trimesh.Scene()
    elif not isinstance(tm_scene, trimesh.Scene):
        raise ValueError(f"Unexpected trimesh data: {type(tm_scene)}")

    # Extract a few things from the scene graph
    # `edges` is a dict {(parent, child): {'geometry': 'name', 'matrix': np.array}}
    edges = tm_scene.graph.transforms.edge_data
    # `leafs` is the list of leaf node names
    leafs = [
        n
        for n in tm_scene.graph.transforms.nodes
        if not tm_scene.graph.transforms.children.get(n, [])
    ]

    # Take care of meshes
    # Note: we're traversing the scene graph only for meshes (i.e. not lights, cameras, etc.)
    # That's because as far as I can tell, trimesh's scene graph only applies to geometries
    # while lights and cameras are stored separate from 'world'.
    if meshes and len(tm_scene.graph.nodes):
        if not flatten:
            # Load the geometries and materials
            gfx_geometries, gfx_materials = objects_from_trimesh(
                tm_scene, materials=materials
            )

            def _build_graph_bfs(node_name, node_object):
                """Recursively parse scene graph into pygfx.Groups."""
                # Note: not sure if that will ever happen in practice but
                # in theory, this implementation could run into the recursion
                # depth limit (3000 on my machine). If that turns out to be
                # a problem we need to switch the implementation

                # Go over this node's children
                for child_name in tm_scene.graph.transforms.children.get(node_name, []):
                    # Is this a leaf node?
                    is_leaf = child_name in leafs

                    # Does it have a geometry?
                    geometry_name = edges[(node_name, child_name)].get("geometry", None)
                    # See if this child has a geometry
                    if geometry_name is not None:
                        # Create the geometry
                        child_object = gfx.Mesh(
                            gfx_geometries[geometry_name],
                            gfx_materials[geometry_name],
                        )
                    else:
                        child_object = gfx.Group()

                    # Apply matrix
                    child_object.local.matrix = edges[(node_name, child_name)]["matrix"]

                    # Connect child to parent unless this is a terminal group
                    if not is_leaf or not isinstance(child_object, gfx.Group):
                        node_object.add(child_object)

                    # If this is not a leaf node, we need to recurse
                    if not is_leaf:
                        _build_graph_bfs(child_name, child_object)

            # Recursively build the scene graph
            world = tm_scene.graph.base_frame
            gfx_scene.local.matrix = tm_scene.graph[world][0]
            _build_graph_bfs(world, gfx_scene)  # start at the root node
        # Just add the flat meshes
        else:
            # If we're flattening the scene graph, we'll just
            # convert all geometries in the scene to meshes
            gfx_scene.add(
                *meshes_from_trimesh(
                    tm_scene, apply_transforms=True, materials=materials
                )
            )

    # Take care of camera
    if camera != "none":
        # Accessing the .camera attribute directly will create
        # a camera if it doesn't exist, hence we check the
        # `.has_camera` attribute.
        # Also note that we're currently only using the
        # camera's transform but not e.g. FOV or resolution:
        if tm_scene.has_camera or camera == "auto":
            gfx_camera = gfx.PerspectiveCamera()
            gfx_scene.add(gfx_camera)

            # If a camera exists in the scene, use its transform
            if tm_scene.has_camera:
                gfx_camera.local.matrix = tm_scene.camera_transform
            # If no camera, make sure the camera actually looks
            # at the objects in the scene (if any)
            elif gfx_scene.get_bounding_box() is not None:
                gfx_camera.show_object(gfx_scene)

    # Take care of lights
    if lights != "none":
        # Parse lights. Similar to the camera, trimesh will create
        # lights automatically if we access the `.lights` attribute
        # directly.
        if hasattr(tm_scene, "_lights"):
            gfx_lights = []
            for light in tm_scene.lights:
                if isinstance(light, trimesh.scene.lighting.PointLight):
                    gfx_lights.append(gfx.PointLight())
                    gfx_lights[-1].distance = light.radius
                elif isinstance(light, trimesh.scene.lighting.DirectionalLight):
                    gfx_lights.append(gfx.DirectionalLight())
                elif isinstance(light, trimesh.scene.lighting.SpotLight):
                    gfx_lights.append(gfx.SpotLight())
                    gfx_lights[-1].distance = light.radius
                    gfx_lights[-1].angle = light.outerConeAngle
                else:
                    # Skip unknown light types
                    continue

                # These properties are common to all light types
                gfx_lights[-1].color = light.color / 255
                gfx_lights[-1].intensity = light.intensity
                gfx_lights[-1].local.matrix = tm_scene.graph[light.name][0]

            gfx_scene.add(*gfx_lights)
        # If no lights (either because the scene did not contain lights or because
        # `lights=False`), make sure things are actually visible
        elif lights == "auto":
            # Add an ambient light
            gfx_scene.add(gfx.AmbientLight())

            # Get the scene bounding box
            bbox = gfx_scene.get_bounding_box()

            # Calculate extent (if no geometries, bbox will be None)
            if bbox is not None:
                extent = bbox[1] - bbox[0]

                # Padd the bounding box
                bbox[0, :] -= extent
                bbox[1, :] += extent

                # Add point lights at the corners of the (padded) bounding box
                # This is ~ what trimesh does
                # If we don't pad, things look odd if the mesh(es) are boxy or flat
                for x, y, z in bbox:
                    light = gfx.PointLight()
                    (light.local.x, light.local.y, light.local.z) = (x, y, z)
                    gfx_scene.add(light)

    # By default trimesh scenes have a white background
    # while gfx.show() uses a black background
    if background:
        gfx_scene.add(gfx.Background(None, gfx.BackgroundMaterial((1, 1, 1))))

    return gfx_scene


def _volume_from_voxelgrid(vxl, cmap=None, clim="data"):
    """Helper function to convert a trimesh VoxelGrid into a pygfx Volume.

    Parameters
    ----------
    vxl : trimesh.voxels.VoxelGrid
        The voxel grid to convert.
    cmap : pygfx.Texture, optional
        The colormap to use. If None, this will default to `pygfx.cm.viridis`.
    clim : "data" | "dtype" | tuple | None
        The contrast limits to scale the data values with. By default ("data"), will
        use the min and max values of the data. If "dtype", will use the theoretical
        limits of the data type. If None, will use [0-1].

    Returns
    -------
    vol : pygfx.Volume
        The volume object.

    """
    import trimesh  # noqa

    if not isinstance(vxl, trimesh.voxel.VoxelGrid):
        raise ValueError(f"Unexpected trimesh data: {type(vxl)}")

    # Extract the matrix. Note that trimesh uses xyz coordinates while
    # pygfx expects zyx - hence we transpose the matrix
    grid = vxl.matrix.T

    # Convert non-native byte order to native; e.g. >u4 -> u4 = uint64
    if grid.dtype.byteorder in (">", "<"):
        grid = grid.astype(grid.dtype.str.replace(grid.dtype.byteorder, ""))
    # Convert boolean matrices to uint16; I tried uint4 but that renders as
    # uniform volume and uint8 looks fuzzy
    elif grid.dtype == bool:
        grid = grid.astype(np.uint16)

    # Initialize texture
    tex = gfx.Texture(grid, dim=3)

    if cmap is None:
        cmap = gfx.cm.cividis

    # Find the potential min/max value of the volume
    if isinstance(clim, str):
        if clim == "data":
            clim = (grid.min(), grid.max())
        elif clim == "dtype":
            clim = (np.iinfo(grid.dtype).max, np.iinfo(grid.dtype).max)

    # Initialize the volume
    vol = gfx.Volume(
        gfx.Geometry(grid=tex),
        gfx.VolumeRayMaterial(clim=clim, map=cmap),
    )

    # Apply transform
    vol.local.matrix = vxl.transform

    return vol
