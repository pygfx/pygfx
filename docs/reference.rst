API Reference
=============

.. rubric:: Sub-Packages

Internally, pygfx is structured into several sub-packages that provide the
functionality exposed in the top-level namespace. At times, you may wish to
search the docs of these sub-packages for additional information. In that case,
you can read more about them here:

.. autosummary::
    :toctree: _autosummary/
    :template: custom_module.rst

    pygfx.cameras
    pygfx.controllers
    pygfx.geometries
    pygfx.helpers
    pygfx.materials
    pygfx.objects
    pygfx.renderers
    pygfx.resources
    pygfx.utils

.. rubric:: Public API

The primary way of accessing pygfx is by using the members of its top-level namespace.
Currently this includes the following classes, which comprise the public API:

.. autosummary::

    ~pygfx.cameras.Camera
    ~pygfx.cameras.NDCCamera
    ~pygfx.cameras.ScreenCoordsCamera
    ~pygfx.cameras.PerspectiveCamera
    ~pygfx.cameras.OrthographicCamera

    ~pygfx.controllers.Controller
    ~pygfx.controllers.PanZoomController
    ~pygfx.controllers.OrbitController
    ~pygfx.controllers.TrackballController
    ~pygfx.controllers.FlyController

    ~pygfx.geometries.box_geometry
    ~pygfx.geometries.cylinder_geometry
    ~pygfx.geometries.cone_geometry
    ~pygfx.geometries.sphere_geometry
    ~pygfx.geometries.plane_geometry
    ~pygfx.geometries.Geometry
    ~pygfx.geometries.geometry_from_trimesh
    ~pygfx.geometries.octahedron_geometry
    ~pygfx.geometries.icosahedron_geometry
    ~pygfx.geometries.dodecahedron_geometry
    ~pygfx.geometries.tetrahedron_geometry
    ~pygfx.geometries.torus_knot_geometry
    ~pygfx.geometries.klein_bottle_geometry
    ~pygfx.helpers.AxesHelper
    ~pygfx.helpers.GridHelper
    ~pygfx.helpers.BoxHelper
    ~pygfx.helpers.TransformGizmo
    ~pygfx.helpers.PointLightHelper
    ~pygfx.helpers.DirectionalLightHelper
    ~pygfx.helpers.SpotLightHelper

    ~pygfx.materials.Material
    ~pygfx.materials.material_from_trimesh
    ~pygfx.materials.MeshAbstractMaterial
    ~pygfx.materials.MeshBasicMaterial
    ~pygfx.materials.MeshPhongMaterial
    ~pygfx.materials.MeshNormalMaterial
    ~pygfx.materials.MeshNormalLinesMaterial
    ~pygfx.materials.MeshSliceMaterial
    ~pygfx.materials.MeshStandardMaterial
    ~pygfx.materials.PointsMaterial
    ~pygfx.materials.PointsGaussianBlobMaterial
    ~pygfx.materials.LineMaterial
    ~pygfx.materials.LineThinMaterial
    ~pygfx.materials.LineThinSegmentMaterial
    ~pygfx.materials.LineSegmentMaterial
    ~pygfx.materials.LineArrowMaterial
    ~pygfx.materials.ImageBasicMaterial
    ~pygfx.materials.VolumeBasicMaterial
    ~pygfx.materials.VolumeSliceMaterial
    ~pygfx.materials.VolumeRayMaterial
    ~pygfx.materials.VolumeMipMaterial
    ~pygfx.materials.BackgroundMaterial
    ~pygfx.materials.BackgroundImageMaterial
    ~pygfx.materials.BackgroundSkyboxMaterial
    ~pygfx.materials.TextMaterial

    ~pygfx.objects.WorldObject
    ~pygfx.objects.Group
    ~pygfx.objects.Scene
    ~pygfx.objects.Background
    ~pygfx.objects.Points
    ~pygfx.objects.Line
    ~pygfx.objects.Mesh
    ~pygfx.objects.Image
    ~pygfx.objects.Volume
    ~pygfx.objects.Text
    ~pygfx.objects.MultiText
    ~pygfx.objects.TextBlock
    ~pygfx.objects.InstancedMesh
    ~pygfx.objects.Light
    ~pygfx.objects.PointLight
    ~pygfx.objects.DirectionalLight
    ~pygfx.objects.AmbientLight
    ~pygfx.objects.SpotLight
    ~pygfx.objects.LightShadow
    ~pygfx.objects.DirectionalLightShadow
    ~pygfx.objects.SpotLightShadow
    ~pygfx.objects.PointLightShadow

    ~pygfx.renderers.Renderer
    ~pygfx.renderers.WgpuRenderer
    ~pygfx.renderers.SvgRenderer

    ~pygfx.resources.Resource
    ~pygfx.resources.Buffer
    ~pygfx.resources.Texture
    ~pygfx.resources.TextureMap

    ~pygfx.utils.color.Color
    ~pygfx.utils.load_gltf.load_gltf
    ~pygfx.utils.load_gltf.load_gltf_async
    ~pygfx.utils.load_gltf.load_gltf_mesh
    ~pygfx.utils.load_gltf.load_gltf_mesh_async
    ~pygfx.utils.load_gltf.print_scene_graph
    ~pygfx.utils.load.load_mesh
    ~pygfx.utils.load.load_meshes
    ~pygfx.utils.load.load_scene
    ~pygfx.utils.show.show
    ~pygfx.utils.show.Display
    ~pygfx.utils.viewport.Viewport
    ~pygfx.utils.text.font_manager
    ~pygfx.utils.cm
    ~pygfx.utils.logger
