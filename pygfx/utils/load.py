"""
Utilities to load scenes from files, using trimesh.
"""

import pygfx as gfx


def load_meshes(path, remote_ok=False):
    """Load meshes from a file.

    Parameters
    ----------
    path : str
        The location where the mesh is stored.
    remote_ok : bool
        Whether to allow loading meshes from URLs, by default False.

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
        Whether to allow loading meshes from URLs, by default False.

    Returns
    -------
    meshes : list
        A list of pygfx.Meshes.

    See Also
    --------
    load_scene
        Loads the entire scene (lights, graph, etc.) from a file.

    """
    import trimesh  # noqa

    # Trimesh's load() performs a similar check and refers
    # loading from URLs to load_remote()
    if "https://" in str(path) or "http://" in str(path):
        if not remote_ok:
            raise ValueError(
                "Loading meshes from URLs is disabled. "
                "Set remote_ok=True to allow loading from URLs."
            )
        scene = trimesh.load_remote(path)
    else:
        scene = trimesh.load(path)

    return meshes_from_trimesh(scene, apply_transforms=True)


def load_scene(path, remote_ok=False):
    """Load file into a scene.

    This function requires the trimesh library.

    Parameters
    ----------
    path : str
        The location where the mesh or scene is stored.
        Can be a local file path or a URL.
    remote_ok : bool
        Whether to allow loading files from URLs, by default False.

    Returns
    -------
    scene : pygfx.Scene
        The loaded scene.

    See Also
    --------
    load_meshes
        Returns the flat meshes contained in a file.

    """
    import trimesh  # noqa

    # Trimesh's load() performs a similar check and refers
    # loading from URLs to load_remote()
    if "https://" in str(path) or "http://" in str(path):
        if not remote_ok:
            raise ValueError(
                "Loading scenes from URLs is disabled. "
                "Set remote_ok=True to allow loading from URLs."
            )
        tm_scene = trimesh.load_remote(path)
    else:
        tm_scene = trimesh.load(path)

    return scene_from_trimesh(tm_scene)


def meshes_from_trimesh(scene, apply_transforms=True):
    """Converts a trimesh scene into a flat list of pygfx Mesh objects.

    Parameters
    ----------
    scene : trimesh.Scene
        The scene to convert.
    apply_transforms : bool
        Whether to apply the scene graph transforms directly to the meshes, by default True.

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
            gfx.material_from_trimesh(scene.visual.material),
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
        gfx_geometries, gfx_materials = objects_from_trimesh(scene)

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


def objects_from_trimesh(scene):
    """Extract geometries and materials from a trimesh scene.

    Parameters
    ----------
    scene : trimesh.Scene
        The scene to convert.

    Returns
    -------
    gfx_geometries : dict
        A dictionary of geometry objects. Keys are the names of the geometries.
    gfx_materials : dict
        A dictionary of material objects. Keys are the names of the geometries.

    """
    gfx_geometries = {}
    gfx_materials = {}
    for name, mesh in scene.geometry.items():
        gfx_geometries[name] = gfx.geometry_from_trimesh(mesh)
        # If mesh has a material, convert it
        if hasattr(mesh.visual, "material"):
            gfx_materials[name] = gfx.material_from_trimesh(mesh.visual.material)
        # If not, create a default material
        else:
            gfx_materials[name] = gfx.MeshStandardMaterial()

    return gfx_geometries, gfx_materials


def scene_from_trimesh(tm_scene):
    """Convert a trimesh scene into a pygfx scene.

    Parameters
    ----------
    tm_scene : trimesh.Scene
        A trimesh Scene.

    Returns
    -------
    scene : pygfx.Scene
        The scene converted to pygfx.

    """
    import trimesh  # noqa

    # Basic scene setup
    gfx_scene = gfx.Scene()
    camera = gfx.PerspectiveCamera()
    gfx_scene.add(camera)

    # This will be populated later
    gfx_lights = []

    if isinstance(tm_scene, trimesh.Trimesh):
        gfx_scene.add(*meshes_from_trimesh(tm_scene))
        camera.show_object(gfx_scene)
    elif isinstance(tm_scene, trimesh.Scene):
        # By convention, we expect the "world" objects to be the root of the scene graph
        G = tm_scene.graph.to_networkx()
        if "world" not in G.nodes:
            raise ValueError("No 'world' node found in scene graph")

        # Load the geometries and materials
        gfx_geometries, gfx_materials = objects_from_trimesh(tm_scene)

        def _build_graph_bfs(node, node_group):
            """Recursively parse scene graph into pygfx.Groups."""
            # Note: not sure if that will ever happen in practice but
            # in theory, this implementation could run into the recursion
            # depth limit (3000 on my machine). If that turns out to be
            # a problem we need to switch the implementation

            # Go over this node's children
            for child in tm_scene.graph.transforms.children.get(node, []):
                # Generate a new graph for this child
                child_group = gfx.Group()

                # Set the child's transform
                child_group.local.matrix = G.edges[(node, child)]["matrix"]

                # See if this child has a geometry
                geometry_name = G.edges[(node, child)].get("geometry", None)
                if geometry_name is not None:
                    # Add the geometry to the child
                    child_group.add(
                        gfx.Mesh(
                            gfx_geometries[geometry_name], gfx_materials[geometry_name]
                        )
                    )

                # Connect the child to the parent
                node_group.add(child_group)

                # Recurse ot this child's children
                _build_graph_bfs(child, child_group)

        # Recursively build the scene graph
        gfx_scene.local.matrix = tm_scene.graph["world"][0]
        _build_graph_bfs("world", gfx_scene)  # start at the root node

        # Accessing the .camera attribute directly will create
        # a camera if it doesn't exist, hence we check the
        # `.has_camera` attribute.
        # Also note that we're currently only using the
        # camera's transform but not e.g. FOV or resolution
        if tm_scene.has_camera:
            camera.local.matrix = tm_scene.camera_transform
        else:
            # If not camera, make sure the camera actually looks
            # at the objects in the scene
            camera.show_object(gfx_scene)

        # Parse lights. Similar to the camera, trimesh will create
        # lights automatically if we access the `.lights` attribute
        # directly.
        if hasattr(tm_scene, "_lights"):
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
    else:
        raise ValueError(f"Unexpected trimesh data: {type(tm_scene)}")

    # If no lights (either because we loaded only a single Trimesh or because
    # the scene didn't contain lights), make sure things are actually visible
    if not gfx_lights:
        # Add an ambient light
        gfx_scene.add(gfx.AmbientLight())

        # Get the scene bounding box
        bbox = gfx_scene.get_bounding_box()

        # Calculate extend (if no geometries, bbox will be None)
        if bbox is not None:
            extend = bbox[1] - bbox[0]

            # Padd the bounding box
            bbox[0, :] -= extend
            bbox[1, :] += extend

            # Add point lights at the corners of the (padded) bounding box
            # This is ~ what trimesh does
            # If we don't pad, things look odd if the mesh(es) are boxy or flat
            for x, y, z in bbox:
                light = gfx.PointLight()
                (light.local.x, light.local.y, light.local.z) = (x, y, z)
                gfx_scene.add(light)

    # By default trimesh scenes have a white background
    # while gfx.show() uses a black background
    gfx_scene.add(gfx.Background(None, gfx.BackgroundMaterial((1, 1, 1))))

    return gfx_scene
