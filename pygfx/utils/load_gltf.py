"""
Utilities to load gltf/glb files, completely compatible with the glTF 2.0 specification.

References:
https://raw.githubusercontent.com/KhronosGroup/glTF/main/specification/2.0/figures/gltfOverview-2.0.0d.png
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

Experimental, not yet fully implemented.
"""

import pygfx as gfx
import numpy as np
import pylinalg as la

from importlib.util import find_spec
from functools import lru_cache


def load_gltf(path, quiet=False, remote_ok=True):
    """
    Load a gltf file and return the content.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    quiet : bool
        Whether to suppress the warning messages.
        Default is False.
    remote_ok : bool
        Whether to allow loading from URLs.
        Default is True.

    Returns:
    ----------
    gltf : object
        The gltf object which contains the following attributes:
        * `scenes`: [gfx.Group]
        * `scene`: gfx.Group or None
        * `cameras`: [gfx.Camera] or None
        * `animations`: [gfx.Animation] or None
    """
    return _GLTF().load(path, quiet, remote_ok)


async def load_gltf_async(path, quiet=False, remote_ok=True):
    """
    Load a gltf file and return the content asynchronously.

    It is recommended to use this function when loading from URLs.
    especially when the gltf model has multiple resources.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    quiet : bool
        Whether to suppress the warning messages.
        Default is False.
    remote_ok : bool
        Whether to allow loading from URLs.
        Default is True.

    Returns:
    ----------
    gltf : object
        The gltf object which contains the following attributes:
        * `scenes`: [gfx.Group]
        * `scene`: gfx.Group or None
        * `cameras`: [gfx.Camera] or None
        * `animations`: [gfx.Animation] or None
    """
    return await _GLTF().load_async(path, quiet, remote_ok)


def load_gltf_mesh(path, materials=True, quiet=False, remote_ok=True):
    """
    Load meshes from a gltf file, without skeletons, and no transformations applied.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    materials : bool
        Whether to load materials.
        Default is True.
    quiet : bool
        Whether to suppress the warning messages.
        Default is False.
    remote_ok : bool
        Whether to allow loading from URLs.
        Default is True.

    Returns:
    ----------
    meshes : list
        A list of pygfx.Meshes.
    """
    return _GLTF().load_mesh(path, quiet, materials=materials, remote_ok=remote_ok)


async def load_gltf_mesh_async(path, materials=True, quiet=False, remote_ok=True):
    """
    Load meshes from a gltf file asynchronously, without skeletons, and no transformations applied.

    It is recommended to use this function when loading from URLs.
    especially when the gltf model has multiple resources.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    materials : bool
        Whether to load materials.
        Default is True.
    quiet : bool
        Whether to suppress the warning messages.
        Default is False.
    remote_ok : bool
        Whether to allow loading from URLs.
        Default is True.

    Returns:
    ----------
    meshes : list
        A list of pygfx.Meshes.
    """
    return await _GLTF().load_mesh_async(
        path, quiet, materials=materials, remote_ok=remote_ok
    )


class _GLTF:
    ACCESSOR_TYPE_SIZE = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT2": 4,
        "MAT3": 9,
        "MAT4": 16,
    }

    COMPONENT_TYPE = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }

    ATTRIBUTE_NAME = {
        "POSITION": "positions",
        "NORMAL": "normals",
        "TANGENT": "tangents",
        "TEXCOORD_0": "texcoords",
        "TEXCOORD_1": "texcoords1",
        "TEXCOORD_2": "texcoords2",
        "TEXCOORD_3": "texcoords3",
        "COLOR_0": "colors",
        "JOINTS_0": "skin_indices",
        "WEIGHTS_0": "skin_weights",
    }

    WRAP_MODE = {
        33071: "clamp-to-edge",  # CLAMP_TO_EDGE
        33648: "mirror-repeat",  # MIRRORED_REPEAT
        10497: "repeat",  # REPEAT
    }

    def __init__(self):
        self.scene = None
        self.scenes = []
        self.cameras = []
        self.lights = []
        self.animations = None

        self._plugins = {}

        self._register_plugin(GLTFMeshQuantizationExtension)
        self._register_plugin(GLTFMaterialsIorExtension)
        self._register_plugin(GLTFMaterialsSpecularExtension)
        self._register_plugin(GLTFMaterialsClearcoatExtension)
        self._register_plugin(GLTFMaterialsIridescenceExtension)
        self._register_plugin(GLTFMaterialsSheenExtension)
        self._register_plugin(GLTFMaterialsAnisotropyExtension)
        self._register_plugin(GLTFMaterialsEmissiveStrengthExtension)
        self._register_plugin(GLTFMaterialsUnlitExtension)
        self._register_plugin(GLTFLightsExtension)
        self._register_plugin(GLTFTextureTransformExtension)

    def _register_plugin(self, plugin_class):
        plugin = plugin_class(self)
        self._plugins[plugin.name] = plugin

    def load(self, path, quiet=False, remote_ok=True):
        """Load the whole gltf file, including meshes, skeletons, cameras, and animations."""
        self.__inner_load(path, quiet, remote_ok)

        self.scenes = self._load_scenes()
        if self._gltf.model.scene is not None:
            self.scene = self.scenes[self._gltf.model.scene]
        if self._gltf.model.animations is not None:
            self.animations = self._load_animations()

        return self

    async def load_async(self, path, quiet=False, remote_ok=True):
        """Load the whole gltf file, including meshes, skeletons, cameras, and animations."""
        await self.__inner_load_async(path, quiet, remote_ok)

        self.scenes = self._load_scenes()
        if self._gltf.model.scene is not None:
            self.scene = self.scenes[self._gltf.model.scene]
        if self._gltf.model.animations is not None:
            self.animations = self._load_animations()

        return self

    def load_mesh(self, path, quiet=False, materials=True, remote_ok=True):
        """Only load meshes from a gltf file, without skeletons, and no transformations applied."""

        self.__inner_load(path, quiet, remote_ok)

        meshes = []
        for gltf_mesh in self._gltf.model.meshes:
            mesh = self._load_gltf_mesh_by_info(gltf_mesh, load_material=materials)
            meshes.extend(mesh)
        return meshes

    async def load_mesh_async(self, path, quiet=False, materials=True, remote_ok=True):
        """Only load meshes from a gltf file, without skeletons, and no transformations applied."""

        await self.__inner_load_async(path, quiet, remote_ok)

        meshes = []
        for gltf_mesh in self._gltf.model.meshes:
            mesh = self._load_gltf_mesh_by_info(gltf_mesh, load_material=materials)
            meshes.extend(mesh)
        return meshes

    def __inner_load(self, path, quiet=False, remote_ok=True):
        if not find_spec("gltflib"):
            raise ImportError(
                "The `gltflib` library is required to load gltf scene: pip install gltflib"
            )
        import gltflib

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

            import httpx
            from io import BytesIO
            from os import path as os_path
            import urllib.parse
            import mimetypes

            # download
            response = httpx.get(path, follow_redirects=True)
            response.raise_for_status()

            file_obj = BytesIO(response.content)

            ext = os_path.splitext(path)[1].lower()
            if ext == ".gltf":
                self._gltf = gltflib.GLTF.read_gltf(file_obj, load_file_resources=False)

            elif ext == ".glb":
                self._gltf = gltflib.GLTF.read_glb(file_obj, load_file_resources=False)

            downloadable_resources = [
                res
                for res in self._gltf.resources
                if isinstance(res, gltflib.FileResource)
            ]

            if downloadable_resources:
                for res in downloadable_resources:
                    res_path = urllib.parse.urljoin(path, res.uri)
                    try:
                        response = httpx.get(res_path)
                        response.raise_for_status()
                        res_file = BytesIO(response.content)

                        res._data = res_file.read()
                        res._mimetype = (
                            res._mimetype or mimetypes.guess_type(res.uri)[0]
                        )
                        res._loaded = True
                    except httpx.HTTPStatusError as e:
                        gfx.utils.logger.warning(f"download failed: {e} - {res_path}")
                    except Exception as e:
                        gfx.utils.logger.warning(f"download failed: {e} - {res_path}")

        else:  # local file
            self._gltf = gltflib.GLTF.load(path, load_file_resources=True)

        self.__post_inner_load(quiet)

    async def __inner_load_async(self, path, quiet=False, remote_ok=True):
        if not find_spec("gltflib"):
            raise ImportError(
                "The `gltflib` library is required to load gltf scene: pip install gltflib"
            )
        import gltflib

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

            import httpx
            from io import BytesIO
            from os import path as os_path
            import urllib.parse
            import mimetypes
            import asyncio

            async with httpx.AsyncClient() as client:
                response = await client.get(path, follow_redirects=True)
                response.raise_for_status()

                file_obj = BytesIO(response.content)

                ext = os_path.splitext(path)[1].lower()
                if ext == ".gltf":
                    self._gltf = gltflib.GLTF.read_gltf(
                        file_obj, load_file_resources=False
                    )

                elif ext == ".glb":
                    self._gltf = gltflib.GLTF.read_glb(
                        file_obj, load_file_resources=False
                    )

                downloadable_resources = [
                    res
                    for res in self._gltf.resources
                    if isinstance(res, gltflib.FileResource)
                ]

                if downloadable_resources:

                    async def download_resource(res, client: httpx.AsyncClient):
                        res_path = urllib.parse.urljoin(path, res.uri)
                        try:
                            response = await client.get(res_path)
                            response.raise_for_status()
                            res_file = BytesIO(response.content)

                            res._data = res_file.read()
                            res._mimetype = (
                                res._mimetype or mimetypes.guess_type(res.uri)[0]
                            )
                            res._loaded = True

                            return res
                        except httpx.HTTPStatusError as e:
                            gfx.utils.logger.warning(
                                f"download failed: {e} - {res_path}"
                            )
                        except Exception as e:
                            gfx.utils.logger.warning(
                                f"download failed: {e} - {res_path}"
                            )

                    tasks = [
                        download_resource(res, client) for res in downloadable_resources
                    ]
                    await asyncio.gather(*tasks)

        else:  # local file
            self._gltf = gltflib.GLTF.load(path, load_file_resources=True)

        self.__post_inner_load(quiet)

    def __post_inner_load(self, quiet):
        if not quiet:
            extensions_required = self._gltf.model.extensionsRequired or []

            supported_extensions = self._plugins.keys()

            unsupported_extensions_required = set(extensions_required) - set(
                supported_extensions
            )

            if unsupported_extensions_required:
                gfx.utils.logger.warning(
                    f"This GLTF required extensions: {unsupported_extensions_required}, which are not supported yet."
                )
                # rasise or ignore?
                # raise NotImplementedError(f"This GLTF required extensions: {extensions_required}, which are not supported yet.")

            extensions_used = self._gltf.model.extensionsUsed or []

            unsupported_extensions_used = set(extensions_used) - set(
                supported_extensions
            )
            if unsupported_extensions_used:
                gfx.utils.logger.warning(
                    f"This GLTF used extensions: {unsupported_extensions_used}, which are not supported yet, so the display may not be so correct."
                )

        # bind the actual data to the buffers
        for buffer in self._gltf.model.buffers:
            buffer.data = self._get_resource_by_uri(buffer.uri).data

        # mark the node types
        self._node_marks = self._mark_nodes()

    @lru_cache(maxsize=None)
    def _get_resource_by_uri(self, uri):
        for resource in self._gltf.resources:
            if resource.uri == uri:
                return resource
        raise ValueError(f"Buffer data not found for buffer {uri}")

    def _mark_nodes(self):
        gltf = self._gltf
        node_marks = [None] * len(gltf.model.nodes)

        # Nothing in the node definition indicates whether it is a Bone.
        # Use the skins' joint references to mark bones.
        if gltf.model.skins:
            for skin in gltf.model.skins:
                for joint in skin.joints:
                    node_marks[joint] = "Bone"

        # Meshes are marked when they are loaded
        # Maybe mark lights and other special nodes here
        return node_marks

    def _load_scenes(self):
        gltf = self._gltf
        scenes = []
        for scene in gltf.model.scenes:
            scene_obj = gfx.Group()
            scene_obj.name = scene.name
            scenes.append(scene_obj)

            for node in scene.nodes:
                node_obj = self._load_node(node)
                scene_obj.add(node_obj)

        return scenes

    @lru_cache(maxsize=None)
    def _load_node(self, node_index):
        gltf = self._gltf
        node_marks = self._node_marks

        node = gltf.model.nodes[node_index]

        translation = node.translation or [0, 0, 0]
        rotation = node.rotation or [0, 0, 0, 1]
        scale = node.scale or [1, 1, 1]

        if node.matrix is not None:
            matrix = np.array(node.matrix).reshape(4, 4).T
        else:
            matrix = la.mat_compose(translation, rotation, scale)

        node_mark = node_marks[node_index]

        if node_mark == "Bone":
            node_obj = gfx.Bone()

        elif node.camera is not None:
            camera_info = gltf.model.cameras[node.camera]
            if camera_info.type == "perspective":
                node_obj = gfx.PerspectiveCamera(
                    camera_info.perspective.yfov,
                    camera_info.perspective.aspectRatio,
                    depth_range=(
                        camera_info.perspective.znear,
                        camera_info.perspective.zfar,
                    ),
                )
            elif camera_info.type == "orthographic":
                node_obj = gfx.OrthographicCamera(
                    camera_info.orthographic.xmag,
                    camera_info.orthographic.ymag,
                    depth_range=(
                        camera_info.orthographic.znear,
                        camera_info.orthographic.zfar,
                    ),
                )
            else:
                raise ValueError(f"Unsupported camera type: {camera_info.type}")

            self.cameras.append(node_obj)
        elif node.mesh is not None:  # Mesh or SkinnedMesh
            # meshes = self._load_gltf_mesh(node.mesh, node.skin)
            # Do not use mesh cache here, we need to create a new mesh object for each node.
            mesh_info = self._gltf.model.meshes[node.mesh]
            meshes = self._load_gltf_mesh_by_info(mesh_info, node.skin)
            if node.weights:
                for gfx_mesh in meshes:
                    gfx_mesh.morph_target_influences = node.weights
            if len(meshes) == 1:
                node_obj = meshes[0]
            else:
                node_obj = gfx.Group()
                for mesh in meshes:
                    node_obj.add(mesh)
        else:
            node_obj = gfx.WorldObject()

        if node.extensions:
            for extension in node.extensions:
                if extension in self._plugins:
                    plugin = self._plugins[extension]
                    if hasattr(plugin, "create_node_attachment"):
                        node_attachment = plugin.create_node_attachment(node_index)
                        if node_attachment is not None:
                            node_obj.add(node_attachment)

        node_obj.local.matrix = matrix
        node_obj.name = node.name

        if node.children:
            for child in node.children:
                child_obj = self._load_node(child)
                node_obj.add(child_obj)

        return node_obj

    def _load_gltf_mesh_by_info(self, mesh, skin_index=None, load_material=True):
        meshes = []
        for primitive in mesh.primitives:
            geometry = self._load_gltf_geometry(primitive)
            primitive_mode = primitive.mode

            if load_material and primitive.material is not None:
                material = self._load_gltf_material(primitive.material)
            else:
                material = gfx.MeshStandardMaterial()
                if hasattr(geometry, "colors"):
                    material.color_mode = "vertex"

            if primitive_mode is None:
                primitive_mode = 4  # default to triangles

            if primitive_mode == 0:
                gfx_mesh = gfx.Points(geometry, material)

            elif primitive_mode in (1, 2, 3):
                # todo: distinguish lines, line_strip, line_loop
                gfx_mesh = gfx.Line(geometry, material)

            elif primitive_mode in (4, 5):
                # todo: distinguish triangles, triangle_strip, triangle_fan
                if skin_index is not None:
                    gfx_mesh = gfx.SkinnedMesh(geometry, material)
                    skeleton = self._load_skins(skin_index)
                    gfx_mesh.bind(skeleton, np.identity(4))
                else:
                    gfx_mesh = gfx.Mesh(geometry, material)
            else:
                raise ValueError(f"Unsupported primitive mode: {primitive.mode}")

            if (
                getattr(geometry, "morph_positions", None)
                or getattr(geometry, "morph_normals", None)
                or getattr(geometry, "morph_colors", None)
            ):
                self.update_morph_target(gfx_mesh, mesh)

            meshes.append(gfx_mesh)

        return meshes

    def update_morph_target(self, gfx_mesh, mesh_def):
        if mesh_def.weights:
            gfx_mesh.morph_target_influences = mesh_def.weights

        if mesh_def.extras and mesh_def.extras.get("targetNames", None):
            gfx_mesh.morph_target_names.extend(mesh_def.extras["targetNames"])

    @lru_cache(maxsize=None)
    def _load_gltf_mesh(self, mesh_index, skin_index=None, load_material=True):
        mesh = self._gltf.model.meshes[mesh_index]
        return self._load_gltf_mesh_by_info(mesh, skin_index, load_material)

    @lru_cache(maxsize=None)
    def _load_gltf_material(self, material_index):
        material = self._gltf.model.materials[material_index]

        extensions = material.extensions or {}

        if (
            GLTFMaterialsUnlitExtension.EXTENSION_NAME in extensions
            and GLTFMaterialsUnlitExtension.EXTENSION_NAME in self._plugins
        ):
            plugin = self._plugins[GLTFMaterialsUnlitExtension.EXTENSION_NAME]
            material_type = plugin.get_material_type(material)
            gfx_material = material_type()
            plugin.extend_material(material, gfx_material)

        else:  # PBR
            material_type = gfx.MeshStandardMaterial

            # check if any plugin can handle this material type
            for extension in extensions:
                if extension in self._plugins:
                    plugin = self._plugins[extension]
                    if hasattr(plugin, "get_material_type"):
                        _material_type = plugin.get_material_type(material)
                        if _material_type is not None:
                            material_type = _material_type
                            break

            gfx_material = material_type()

            # check if any plugin can extend the material
            for extension in extensions:
                if extension in self._plugins:
                    plugin = self._plugins[extension]
                    if hasattr(plugin, "extend_material"):
                        plugin.extend_material(material, gfx_material)

            pbr_metallic_roughness = material.pbrMetallicRoughness
            if pbr_metallic_roughness is not None:
                if pbr_metallic_roughness.baseColorFactor is not None:
                    gfx_material.color = gfx.Color.from_physical(
                        *pbr_metallic_roughness.baseColorFactor
                    )

                if pbr_metallic_roughness.baseColorTexture is not None:
                    gfx_material.map = self._load_gltf_texture_map(
                        pbr_metallic_roughness.baseColorTexture
                    )

                if pbr_metallic_roughness.metallicRoughnessTexture is not None:
                    metallic_roughness_map = self._load_gltf_texture_map(
                        pbr_metallic_roughness.metallicRoughnessTexture
                    )
                    gfx_material.roughness_map = metallic_roughness_map
                    gfx_material.metalness_map = metallic_roughness_map

                if pbr_metallic_roughness.roughnessFactor is not None:
                    gfx_material.roughness = pbr_metallic_roughness.roughnessFactor
                else:
                    gfx_material.roughness = 1.0

                if pbr_metallic_roughness.metallicFactor is not None:
                    gfx_material.metalness = pbr_metallic_roughness.metallicFactor
                else:
                    gfx_material.metalness = 1.0

            if material.normalTexture is not None:
                gfx_material.normal_map = self._load_gltf_texture_map(
                    material.normalTexture
                )
                scale_factor = material.normalTexture.scale
                if scale_factor is None:
                    scale_factor = 1.0

                gfx_material.normal_scale = (scale_factor, scale_factor)

            if material.occlusionTexture is not None:
                gfx_material.ao_map = self._load_gltf_texture_map(
                    material.occlusionTexture
                )

            if material.emissiveFactor is not None:
                gfx_material.emissive = gfx.Color.from_physical(
                    *material.emissiveFactor
                )

            if material.emissiveTexture is not None:
                gfx_material.emissive_map = self._load_gltf_texture_map(
                    material.emissiveTexture
                )

        # todo alphaMode
        # todo alphaCutoff

        gfx_material.side = (
            gfx.enums.VisibleSide.both
            if material.doubleSided
            else gfx.enums.VisibleSide.front
        )

        return gfx_material

    def _load_gltf_texture_map(self, texture_info):
        if isinstance(texture_info, dict):
            texture_index = texture_info["index"]
            uv_channel = texture_info.get("texCoord", 0)
            extensions = texture_info.get("extensions", {})
        else:
            texture_index = texture_info.index
            uv_channel = texture_info.texCoord or 0
            extensions = texture_info.extensions or {}

        texture = self._load_gltf_texture(texture_index)

        texture_map = gfx.TextureMap(texture, uv_channel=uv_channel)

        texture_desc = self._gltf.model.textures[texture_index]

        sampler = texture_desc.sampler

        if sampler is not None:
            sampler = self._load_gltf_sampler(sampler)

            if sampler.magFilter == 9728:
                texture_map.mag_filter = "nearest"
            elif sampler.magFilter == 9729 or sampler.magFilter is None:
                texture_map.mag_filter = "linear"

            if sampler.minFilter == 9728:  # NEAREST
                texture_map.min_filter = "nearest"
            elif sampler.minFilter == 9729:  # LINEAR
                texture_map.min_filter = "linear"
            elif sampler.minFilter == 9984:  # NEAREST_MIPMAP_NEAREST
                texture._generate_mipmaps = True
                texture_map.min_filter = "nearest"
                texture_map.mipmap_filter = "nearest"
            elif sampler.minFilter == 9985:  # LINEAR_MIPMAP_NEAREST
                texture._generate_mipmaps = True
                texture_map.min_filter = "linear"
                texture_map.mipmap_filter = "nearest"
            elif sampler.minFilter == 9986:  # NEAREST_MIPMAP_LINEAR
                texture._generate_mipmaps = True
                texture_map.min_filter = "nearest"
                texture_map.mipmap_filter = "linear"
            elif (
                sampler.minFilter == 9987 or sampler.minFilter is None
            ):  # LINEAR_MIPMAP_LINEAR
                texture._generate_mipmaps = True
                texture_map.min_filter = "linear"
                texture_map.mipmap_filter = "linear"

            texture_map.wrap_s = self.WRAP_MODE[sampler.wrapS or 10497]
            texture_map.wrap_t = self.WRAP_MODE[sampler.wrapT or 10497]

        for extension in extensions:
            if extension in self._plugins:
                plugin = self._plugins[extension]
                if hasattr(plugin, "extend_texture"):
                    plugin.extend_texture(texture_info, texture_map)

        return texture_map

    @lru_cache(maxsize=None)
    def _load_gltf_texture(self, texture_index):
        texture_desc = self._gltf.model.textures[texture_index]
        source = texture_desc.source
        image = self._load_image(source)
        texture = gfx.Texture(image, dim=2)
        return texture

    @lru_cache(maxsize=None)
    def _load_gltf_sampler(self, sampler_index):
        sampler = self._gltf.model.samplers[sampler_index]
        # Sampler( magFilter=9729, minFilter=9987, wrapS=None, wrapT=None)
        return sampler

    @lru_cache(maxsize=None)
    def _load_image(self, image_index):
        image_info = self._gltf.model.images[image_index]
        import imageio.v3 as iio

        if image_info.bufferView is not None:
            image_data = self._get_buffer_memory_view(image_info.bufferView)

        elif image_info.uri is not None:
            resource = self._get_resource_by_uri(image_info.uri)
            image_data = resource.data
        else:
            raise ValueError("No image data found")

        # need consider mimeType?
        image = iio.imread(image_data, pilmode="RGBA")
        return image

    def _load_gltf_geometry(self, primitive):
        indices_accessor = primitive.indices

        geometry_args = {}

        for attr, accessor_index in primitive.attributes.__dict__.items():
            if accessor_index is not None:
                geometry_attr = self.ATTRIBUTE_NAME[attr]
                data = self._load_accessor(accessor_index)

                # pygfx not support int attributes now, so we need to convert them to float.
                if geometry_attr in (
                    "positions",
                    "normals",
                    "tangents",
                    "texcoords",
                    "texcoords1",
                    "texcoords2",
                    "texcoords3",
                ):
                    data = data.astype(np.float32, copy=False)

                geometry_args[geometry_attr] = data

        if indices_accessor is not None:
            indices = self._load_accessor(indices_accessor).reshape(-1, 3)
        else:
            # TODO: For now, pygfx not support non-indexed geometry, so we need to generate indices for them.
            # Remove this after pygfx support non-indexed geometry.
            indices = np.arange(
                len(geometry_args["positions"]) // 3 * 3, dtype=np.int32
            ).reshape((-1, 3))

        geometry_args["indices"] = indices

        geometry = gfx.Geometry(**geometry_args)

        if primitive.targets:
            for target in primitive.targets:
                # print(target)
                for attr, accessor_index in target.__dict__.items():
                    if accessor_index is not None:
                        target_attr = f"morph_{self.ATTRIBUTE_NAME[attr]}"
                        data = self._load_accessor(accessor_index).astype(
                            np.float32, copy=False
                        )

                        morph_attr = getattr(geometry, target_attr, None)
                        if morph_attr is None:
                            morph_attr = []
                            setattr(geometry, target_attr, morph_attr)

                        morph_attr.append(data)

            geometry.morph_targets_relative = True

        return geometry

    @lru_cache(maxsize=None)
    def _get_buffer_memory_view(self, buffer_view_index):
        gltf = self._gltf
        buffer_view = gltf.model.bufferViews[buffer_view_index]
        buffer = gltf.model.buffers[buffer_view.buffer]
        m = memoryview(buffer.data)
        view = m[
            buffer_view.byteOffset : (buffer_view.byteOffset or 0)
            + buffer_view.byteLength
        ]
        return view

    @lru_cache(maxsize=None)
    def _load_accessor(self, accessor_index):
        gltf = self._gltf
        accessor = gltf.model.accessors[accessor_index]

        buffer_view = gltf.model.bufferViews[accessor.bufferView]
        view = self._get_buffer_memory_view(accessor.bufferView)

        # todo accessor.sparse
        if accessor.sparse is not None:
            gfx.utils.logger.warning("Sparse accessor is not supported yet.")

        accessor_type = accessor.type
        accessor_component_type = accessor.componentType
        accessor_count = accessor.count
        accessor_dtype = np.dtype(self.COMPONENT_TYPE[accessor_component_type])
        accessor_offset = accessor.byteOffset or 0
        accessor_type_size = self.ACCESSOR_TYPE_SIZE[accessor_type]

        if buffer_view.byteStride is not None:
            # It's a interleaved buffer
            # pygfx not support interleaved buffer now, so we pick out the data we need from the interleaved buffer.
            # TODO: optimize this after pygfx support interleaved buffer.
            ar = np.lib.stride_tricks.as_strided(
                view[accessor_offset:],
                shape=(accessor_count, accessor_type_size * accessor_dtype.itemsize),
                strides=(buffer_view.byteStride, 1),
            )
            ar = np.frombuffer(np.ascontiguousarray(ar), dtype=accessor_dtype)
        else:
            ar = np.frombuffer(
                view,
                dtype=accessor_dtype,
                offset=accessor_offset,
                count=accessor_count * accessor_type_size,
            )
        if accessor_type_size > 1:
            ar = ar.reshape(accessor_count, accessor_type_size)

        if accessor.normalized:
            # KHR_mesh_quantization
            # https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_mesh_quantization
            assert accessor_dtype.kind == "i" or accessor_dtype.kind == "u"
            ar = ar.astype(np.float32, copy=False) / np.iinfo(accessor_dtype).max

        # pygfx not support int8, int16, uint8, uint16 now
        if ar.dtype == np.uint8 or ar.dtype == np.uint16:
            ar = ar.astype(np.uint32, copy=False)
        if ar.dtype == np.int8 or ar.dtype == np.int16:
            ar = ar.astype(np.int32, copy=False)

        return ar

    @lru_cache(maxsize=None)
    def _load_skins(self, skin_index):
        skin = self._gltf.model.skins[skin_index]
        bones = [self._load_node(index) for index in skin.joints]
        inverse_bind_matrices = self._load_accessor(skin.inverseBindMatrices)

        bone_inverses = []
        for matrices in inverse_bind_matrices:
            bone_inverse = np.array(matrices).reshape(4, 4).T
            bone_inverses.append(bone_inverse)

        skeleton = gfx.Skeleton(bones, bone_inverses)
        return skeleton

    def _load_animation(self, animation_info):
        channels = animation_info.channels
        samplers = animation_info.samplers

        duration = 0

        key_frame_tracks = []

        for channel in channels:
            target = channel.target
            sampler = samplers[channel.sampler]

            if target.node is None:
                # todo: now we only support node animation
                gfx.utils.logger.warning(
                    f"Animation channel target [{target}] node is None, skip it for now."
                )
                continue

            target_node = self._load_node(target.node)
            name = target_node.name
            target_property = target.path
            interpolation = sampler.interpolation or "LINEAR"
            times = self._load_accessor(sampler.input)
            if times[-1] > duration:
                duration = times[-1]
            values = self._load_accessor(sampler.output)

            if interpolation == "LINEAR":
                if target_property == "rotation":
                    interpolation_fn = gfx.QuaternionLinearInterpolant
                else:
                    interpolation_fn = gfx.LinearInterpolant
            elif interpolation == "STEP":
                interpolation_fn = gfx.StepInterpolant
            elif interpolation == "CUBICSPLINE":
                # interpolation_fn = gfx.CubicSplineInterpolant
                # A CUBICSPLINE keyframe in glTF has three output values for each input value,
                # representing inTangent, splineVertex, and outTangent.
                interpolation_fn = (
                    GLTFCubicSplineInterpolant
                    if target_property != "rotation"
                    else GLTFCubicSplineQuaternionInterpolant
                )
            else:
                raise ValueError(f"Unsupported interpolation type: {interpolation}")

            if interpolation == "CUBICSPLINE":
                # Layout of keyframe output values for CUBICSPLINE animations:
                # [ inTangent_1, splineVertex_1, outTangent_1, inTangent_2, splineVertex_2, ... ]
                values = values.reshape(len(times), -1, values.shape[-1])
            else:
                values = values.reshape(len(times), -1)

            if len(times) != len(values):
                gfx.utils.logger.warning(
                    f"keyframe: {name}, times and values have different lengths, {len(times)} != {len(values)}"
                )
                length = min(len(times), len(values))
                times = times[:length]
                values = values[:length]

            keyframe = gfx.KeyframeTrack(
                name, target_node, target_property, times, values, interpolation_fn
            )
            key_frame_tracks.append(keyframe)

        action_clip = gfx.AnimationClip(animation_info.name, duration, key_frame_tracks)

        return action_clip

    def _load_animations(self):
        gltf = self._gltf
        animations = []
        for animation in gltf.model.animations:
            action_clip = self._load_animation(animation)
            animations.append(action_clip)
        return animations


def print_scene_graph(obj, show_pos=False, show_rot=False, show_scale=False):
    """
    Print the tree structure of the scene, including the optional position, rotation, and scale of each object.

    Parameters:
    ----------
    obj : gfx.WorldObject
        The root object.
    show_pos : bool
        Whether to show the position of each object.
    show_rot : bool
        Whether to show the rotation of each object.
    show_scale : bool
        Whether to show the scale of each object.
    """

    def _print_tree(obj: gfx.WorldObject, level=0):
        name = "- " * level + f"{obj.__class__.__name__}[{obj.name}]"
        if show_pos:
            name += f"\n{'  ' * level}|- pos: {obj.local.position}"
        if show_rot:
            name += f"\n{'  ' * level}|- rot: {obj.local.rotation}"
        if show_scale:
            name += f"\n{'  ' * level}|- scale: {obj.local.scale}"

        print(name)

        for child in obj.children:
            _print_tree(child, level=level + 1)

    _print_tree(obj)


class GLTFCubicSplineInterpolant(gfx.Interpolant):
    """
    See: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
    """

    def __init__(self, times, values):
        super().__init__(times, values)

    def _interpolate(self, i1, t0, t, t1):
        dt = t1 - t0

        p = (t - t0) / dt
        pp = p * p
        ppp = pp * p

        s2 = -2 * ppp + 3 * pp
        s3 = ppp - pp
        s0 = 1 - s2
        s1 = s3 - pp + p

        # Layout of keyframe output values for CUBICSPLINE animations:
        # [ [inTangent_1, splineVertex_1, outTangent_1], [inTangent_2, splineVertex_2, outTangent_2], ... ]

        values = self.sample_values
        p0 = values[i1 - 1][1]
        m0 = values[i1 - 1][2] * dt

        p1 = values[i1][1]
        m1 = values[i1][0] * dt

        return s0 * p0 + s1 * m0 + s2 * p1 + s3 * m1


class GLTFCubicSplineQuaternionInterpolant(GLTFCubicSplineInterpolant):
    def _interpolate(self, i1, t0, t, t1):
        res = super()._interpolate(i1, t0, t, t1)
        # remember normalize the quaternion
        return res / np.linalg.norm(res)


### GLTF Extensions ###


class GLTFExtension:
    """
    Abstract Base class for GLTF extensions.
    """

    EXTENSION_NAME = ""

    def __init__(self, parser: _GLTF):
        self.name = self.EXTENSION_NAME
        self.parser = parser


class GLTFMeshQuantizationExtension(GLTFExtension):
    EXTENSION_NAME = "KHR_mesh_quantization"


class GLTFBaseMaterialsExtension(GLTFExtension):
    """
    Abstract Base class for GLTF materials extensions.
    """

    MATERIAL_TYPE = gfx.MeshPhysicalMaterial

    def get_material_type(self, material_def):
        if material_def.extensions and self.EXTENSION_NAME in material_def.extensions:
            return self.MATERIAL_TYPE
        else:
            return None

    def extend_material(self, material_def, material):
        pass


class GLTFMaterialsIorExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_ior"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        material.ior = extension.get("ior", 1.5)


class GLTFMaterialsSpecularExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_specular"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        material.specular_intensity = extension.get("specularFactor", 1.0)

        specular_texture = extension.get("specularTexture", None)

        if specular_texture is not None:
            material.specular_intensity_map = self.parser._load_gltf_texture_map(
                specular_texture
            )

        specular_color = extension.get("specularColorFactor", [1.0, 1.0, 1.0])
        material.specular = gfx.Color.from_physical(*specular_color)

        specular_color_texture = extension.get("specularColorTexture", None)

        if specular_color_texture is not None:
            material.specular_map = self.parser._load_gltf_texture_map(
                specular_color_texture
            )


class GLTFMaterialsClearcoatExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_clearcoat"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        clearcoat_factor = extension.get("clearcoatFactor", None)
        if clearcoat_factor is not None:
            material.clearcoat = clearcoat_factor

        clearcoat_texture = extension.get("clearcoatTexture", None)
        if clearcoat_texture is not None:
            material.clearcoat_map = self.parser._load_gltf_texture_map(
                clearcoat_texture
            )

        clearcoat_roughness_factor = extension.get("clearcoatRoughnessFactor", None)
        if clearcoat_roughness_factor is not None:
            material.clearcoat_roughness = clearcoat_roughness_factor

        clearcoat_rughness_texture = extension.get("clearcoatRoughnessTexture", None)
        if clearcoat_rughness_texture is not None:
            material.clearcoat_roughness_map = self.parser._load_gltf_texture_map(
                clearcoat_rughness_texture
            )

        clearcoat_normal_texture = extension.get("clearcoatNormalTexture", None)
        if clearcoat_normal_texture is not None:
            material.clearcoat_normal_map = self.parser._load_gltf_texture_map(
                clearcoat_normal_texture
            )
            clearcoat_normal_scale = clearcoat_normal_texture.get("scale", None)
            if clearcoat_normal_scale is not None:
                material.clearcoat_normal_scale = (
                    clearcoat_normal_scale,
                    clearcoat_normal_scale,
                )


class GLTFMaterialsIridescenceExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_iridescence"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        iridescence_factor = extension.get("iridescenceFactor", None)
        if iridescence_factor is not None:
            material.iridescence = iridescence_factor

        iridescence_texture = extension.get("iridescenceTexture", None)
        if iridescence_texture is not None:
            material.iridescence_map = self.parser._load_gltf_texture_map(
                iridescence_texture
            )

        iridescence_ior = extension.get("iridescenceIor", None)
        material.iridescence_ior = iridescence_ior or 1.3

        iridescence_thickness_min = extension.get("iridescenceThicknessMinimum", 100)
        iridescence_thickness_max = extension.get("iridescenceThicknessMaximum", 400)

        material.iridescence_thickness_range = (
            iridescence_thickness_min,
            iridescence_thickness_max,
        )

        iridescence_thickness_texture = extension.get(
            "iridescenceThicknessTexture", None
        )
        if iridescence_thickness_texture is not None:
            material.iridescence_thickness_map = self.parser._load_gltf_texture_map(
                iridescence_thickness_texture
            )


class GLTFMaterialsAnisotropyExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_anisotropy"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        anisotropy_factor = extension.get("anisotropyStrength", None)
        if anisotropy_factor is not None:
            material.anisotropy = anisotropy_factor

        anisotropy_rotation = extension.get("anisotropyRotation", None)
        if anisotropy_rotation is not None:
            material.anisotropy_rotation = anisotropy_rotation

        anisotropy_texture = extension.get("anisotropyTexture", None)
        if anisotropy_texture is not None:
            material.anisotropy_map = self.parser._load_gltf_texture_map(
                anisotropy_texture
            )


class GLTFMaterialsSheenExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_sheen"

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]

        sheen_color = extension.get("sheenColorFactor", None)
        if sheen_color is not None:
            material.sheen = 1.0
            material.sheen_color = gfx.Color.from_physical(*sheen_color)

        sheen_color_texture = extension.get("sheenColorTexture", None)
        if sheen_color_texture is not None:
            material.sheen_color_map = self._load_texture(sheen_color_texture)

        sheen_roughness_factor = extension.get("sheenRoughnessFactor", None)
        if sheen_roughness_factor is not None:
            material.sheen_roughness = sheen_roughness_factor

        sheen_roughness_texture = extension.get("sheenRoughnessTexture", None)
        if sheen_roughness_texture is not None:
            material.sheen_roughness_map = self._load_texture(sheen_roughness_texture)


class GLTFMaterialsEmissiveStrengthExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_emissive_strength"
    MATERIAL_TYPE = None

    def get_material_type(self, material_def):
        return None

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        extension = material_def.extensions[self.EXTENSION_NAME]
        emissive_strength = extension.get("emissiveStrength", None)
        if emissive_strength is not None:
            material.emissive_intensity = emissive_strength


class GLTFMaterialsUnlitExtension(GLTFBaseMaterialsExtension):
    EXTENSION_NAME = "KHR_materials_unlit"
    MATERIAL_TYPE = gfx.MeshBasicMaterial

    def extend_material(self, material_def, material):
        if (
            not material_def.extensions
            or self.EXTENSION_NAME not in material_def.extensions
        ):
            return

        pbr_metallic_roughness = material_def.pbrMetallicRoughness
        if pbr_metallic_roughness is not None:
            if pbr_metallic_roughness.baseColorFactor is not None:
                material.color = gfx.Color.from_physical(
                    *pbr_metallic_roughness.baseColorFactor
                )

            if pbr_metallic_roughness.baseColorTexture is not None:
                material.map = self.parser._load_gltf_texture_map(
                    pbr_metallic_roughness.baseColorTexture
                )


class GLTFLightsExtension(GLTFExtension):
    EXTENSION_NAME = "KHR_lights_punctual"

    def create_node_attachment(self, node_index):
        """
        Create a light object and attach it to the node.
        """
        gltf = self.parser._gltf
        node = gltf.model.nodes[node_index]
        if node.extensions and self.EXTENSION_NAME in node.extensions:
            extension = node.extensions[self.EXTENSION_NAME]
            light_index = extension["light"]

            light = self._load_light(light_index)

            if light is not None:
                self.parser.lights.append(light)

            return light

    @lru_cache(maxsize=None)
    def _load_light(self, light_index):
        gltf = self.parser._gltf

        extensions = (gltf.model.extensions and gltf.model.extensions[self.name]) or {}
        light_defs = extensions.get("lights", None)

        if light_defs:
            if light_index >= len(light_defs):
                return None

            light_info = light_defs[light_index]

            color = light_info.get("color", [1, 1, 1])
            light_color = gfx.Color.from_physical(*color)
            light_range = light_info.get("range", 0)
            intensity = light_info.get("intensity", 1.0)

            light_type = light_info["type"]

            if light_type == "directional":
                light = gfx.DirectionalLight(color=light_color, intensity=intensity)
                light.target.local.position = (0, 0, -1)
                light.add(light.target)
            elif light_type == "point":
                light = gfx.PointLight(color=light_color, intensity=intensity)
                light.distance = light_range
            elif light_type == "spot":
                light = gfx.SpotLight(color=light_color, intensity=intensity)
                light.distance = light_range

                if "spot" in light_info:
                    spot = light_info["spot"]
                    inner_cone_angle = spot.get("innerConeAngle", 0)
                    outer_cone_angle = spot.get("innerConeAngle", np.pi / 4)
                    light.angle = outer_cone_angle
                    light.penumbra = 1.0 - (inner_cone_angle / outer_cone_angle)
                    light.target.local.position = (0, 0, -1)
                    light.add(light.target)

            else:
                raise ValueError(f"Unsupported light type: {light_info.type}")

            light.local.position = (0, 0, 0)
            light.name = light_info.get("name", f"light_{light_index}")

            return light


class GLTFTextureTransformExtension(GLTFExtension):
    EXTENSION_NAME = "KHR_texture_transform"

    def extend_texture(self, texture_info, texture_map):
        if isinstance(texture_info, dict):
            extensions = texture_info.get("extensions", None)
        else:
            extensions = texture_info.extensions

        if self.EXTENSION_NAME not in extensions:
            return

        extension = extensions[self.EXTENSION_NAME]

        texture_map.update_matrix_components(
            offset=extension.get("offset", None),
            rotation=extension.get("rotation", None),
            scale=extension.get("scale", None),
        )

        texcoord = extension.get("texCoord", None)
        if texcoord is not None:
            texture_map.uv_channel = texcoord
