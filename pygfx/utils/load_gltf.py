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


def load_gltf(path, quiet=False):
    """
    Load a gltf file and return the content.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    quiet : bool
        Whether to suppress the warning messages.

    Returns:
    ----------
    gltf : object
        The gltf object which contains the following attributes:
        * `scenes`: [gfx.Group]
        * `scene`: gfx.Group or None
        * `cameras`: [gfx.Camera] or None
        * `animations`: [gfx.Animation] or None
    """
    return _GLTF(path, quiet).load()


def load_gltf_mesh(path, materials=True, quiet=False):
    """
    Load meshes from a gltf file, without skeletons, and no transformations applied.

    This function requires the gltflib library.

    Parameters:
    ----------
    path : str
        The path to the gltf file.
    materials : bool
        Whether to load materials.
    quiet : bool
        Whether to suppress the warning messages.

    Returns:
    ----------
    meshes : list
        A list of pygfx.Meshes.
    """
    return _GLTF(path, quiet).load_mesh(materials=materials)


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
        "COLOR_0": "colors",
        "JOINTS_0": "skin_indices",
        "WEIGHTS_0": "skin_weights",
    }

    SUPPORTED_EXTENSIONS = ["KHR_mesh_quantization"]

    def __init__(self, path, quiet=False):
        self._path = path
        self.scene = None
        self.scenes = []
        self.cameras = None
        self.animations = None

        self.__inner_load(quiet)

    def load(self):
        """Load the whole gltf file, including meshes, skeletons, cameras, and animations."""

        self.scenes = self._load_scenes()
        if self._gltf.model.scene is not None:
            self.scene = self.scenes[self._gltf.model.scene]
        if self._gltf.model.animations is not None:
            self.animations = self._load_animations()

        # TODO:
        # self.cameras
        return self

    def load_mesh(self, materials=True):
        """Only load meshes from a gltf file, without skeletons, and no transformations applied."""

        meshes = []
        for gltf_mesh in self._gltf.model.meshes:
            mesh = self._load_gltf_mesh_by_info(gltf_mesh, load_material=materials)
            meshes.extend(mesh)
        return meshes

    def __inner_load(self, quiet=False):
        if not find_spec("gltflib"):
            raise ImportError(
                "The `gltflib` library is required to load gltf scene: pip install gltflib"
            )
        import gltflib  # noqa

        path = self._path
        self._gltf = gltflib.GLTF.load(path, load_file_resources=True)

        if not quiet:
            extensions_required = self._gltf.model.extensionsRequired or []

            unsupported_extensions_required = set(extensions_required) - set(
                self.SUPPORTED_EXTENSIONS
            )

            if unsupported_extensions_required:
                gfx.utils.logger.warning(
                    f"This GLTF required extensions: {unsupported_extensions_required}, which are not supported yet."
                )
                # rasise or ignore?
                # raise NotImplementedError(f"This GLTF required extensions: {extensions_required}, which are not supported yet.")

            extensions_used = self._gltf.model.extensionsUsed or []

            unsupported_extensions_used = set(extensions_used) - set(
                self.SUPPORTED_EXTENSIONS
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

        # Mark cameras
        if gltf.model.cameras:
            for camera in gltf.model.cameras:
                node_marks[camera.node] = "Camera"

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

        if node.matrix is not None:
            matrix = np.array(node.matrix).reshape(4, 4).T
        else:
            translation = node.translation or [0, 0, 0]
            rotation = node.rotation or [0, 0, 0, 1]
            scale = node.scale or [1, 1, 1]
            matrix = la.mat_compose(translation, rotation, scale)

        node_mark = node_marks[node_index]

        if node_mark == "Bone":
            node_obj = gfx.Bone()
        elif node_mark == "Camera":
            # TODO: implement camera loading
            # node_obj = gfx.Camera()
            pass
        elif node.mesh is not None:  # Mesh or SkinnedMesh
            meshes = self._load_gltf_mesh(node.mesh, node.skin)
            if len(meshes) == 1:
                node_obj = meshes[0]
            else:
                node_obj = gfx.Group()
                for mesh in meshes:
                    node_obj.add(mesh)
        else:
            node_obj = gfx.WorldObject()

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

            if primitive_mode is None:
                primitive_mode = 4  # default to triangles

            if primitive_mode == 0:
                material = gfx.PointsMaterial()
                gfx_mesh = gfx.Points(geometry, material)

            elif primitive_mode in (1, 2, 3):
                material = gfx.LineSegmentMaterial()
                gfx_mesh = gfx.Line(geometry, material)

            elif primitive_mode in (4, 5):
                if load_material and primitive.material is not None:
                    material = self._load_gltf_material(primitive.material)
                else:
                    material = gfx.MeshBasicMaterial()

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
        pbr_metallic_roughness = material.pbrMetallicRoughness

        if pbr_metallic_roughness is not None:
            gfx_material = gfx.MeshStandardMaterial()

            if pbr_metallic_roughness.baseColorTexture is not None:
                gfx_material.map = self._load_gltf_texture(
                    pbr_metallic_roughness.baseColorTexture
                )

            if pbr_metallic_roughness.metallicRoughnessTexture is not None:
                metallic_roughness_map = self._load_gltf_texture(
                    pbr_metallic_roughness.metallicRoughnessTexture
                )
                gfx_material.roughness_map = metallic_roughness_map
                gfx_material.metalness_map = metallic_roughness_map
                gfx_material.roughness = 1.0
                gfx_material.metalness = 1.0

            if pbr_metallic_roughness.roughnessFactor is not None:
                gfx_material.roughness = pbr_metallic_roughness.roughnessFactor

            if pbr_metallic_roughness.metallicFactor is not None:
                gfx_material.metalness = pbr_metallic_roughness.metallicFactor

        else:
            gfx_material = gfx.MeshPhongMaterial()

        if material.normalTexture is not None:
            gfx_material.normal_map = self._load_gltf_texture(material.normalTexture)
            scale_factor = material.normalTexture.scale
            if scale_factor is None:
                scale_factor = 1.0

            # pygfx now assume the normal map is in tangent space, so we need to flip the y-axis.
            # See: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
            # TODO: support object space normal map, and flip the y-axis when only in tangent space.(check if the tangent attribute in the mesh primitive)
            gfx_material.normal_scale = (scale_factor, -scale_factor)

        if material.occlusionTexture is not None:
            gfx_material.ao_map = self._load_gltf_texture(material.occlusionTexture)

        if material.emissiveFactor is not None:
            gfx_material.emissive = gfx.Color(*material.emissiveFactor)

        if material.emissiveTexture is not None:
            gfx_material.emissive_map = self._load_gltf_texture(
                material.emissiveTexture
            )

        return gfx_material

    def _load_gltf_texture(self, texture_info):
        texture_index = texture_info.index
        texture = self._load_gltf_texture_resource(texture_index)
        # uv_channel = texture_info.texCoord
        # TODO: use uv_channel when pygfx supports it
        return texture

    @lru_cache(maxsize=None)
    def _load_gltf_texture_resource(self, texture_index):
        texture_desc = self._gltf.model.textures[texture_index]
        source = texture_desc.source
        image = self._load_image(source)
        texture = gfx.Texture(image, dim=2)

        sampler = texture_desc.sampler
        sampler = self._load_gltf_sampler(sampler)
        # pygfx not support set texture sampler info now
        # TODO: implement this after pygfx support texture custom sampler
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
        image = iio.imread(image_data)
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
                ):
                    data = data.astype(np.float32, copy=False)

                geometry_args[geometry_attr] = data

        if indices_accessor is not None:
            indices = self._load_accessor(indices_accessor).reshape(-1, 3)
        else:
            # TODO: For now, pygfx not support non-indexed geometry, so we need to generate indices for them.
            # Remove this after pygfx support non-indexed geometry.
            indices = np.arange(
                len(geometry_args["positions"]) * 3, dtype=np.int32
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
            buffer_view.byteOffset : buffer_view.byteOffset + buffer_view.byteLength
        ]
        return view

    @lru_cache(maxsize=None)
    def _load_accessor(self, accessor_index):
        gltf = self._gltf
        accessor = gltf.model.accessors[accessor_index]

        buffer_view = gltf.model.bufferViews[accessor.bufferView]
        view = self._get_buffer_memory_view(accessor.bufferView)

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
            ar = np.frombuffer(np.ascontiguousarray(ar), dtype=accessor_dtype).reshape(
                accessor_count, accessor_type_size
            )
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

            target_node = self._load_node(target.node)
            name = target_node.name
            target_property = target.path
            interpolation = sampler.interpolation
            times = self._load_accessor(sampler.input)
            if times[-1] > duration:
                duration = times[-1]
            values = self._load_accessor(sampler.output)

            key_frame_tracks.append(
                {
                    "name": name,
                    "target": target_node,
                    "property": target_property,
                    "interpolation": interpolation,
                    "times": times,
                    "values": values,
                }
            )

        action_clip = {
            "name": animation_info.name,
            "duration": duration,
            "tracks": key_frame_tracks,
        }

        return action_clip

    def _load_animations(self):
        gltf = self._gltf
        animations = []
        for animation in gltf.model.animations:
            action_clip = self._load_animation(animation)
            animations.append(action_clip)
        return animations


def print_tree(obj, show_pos=False, show_rot=False, show_scale=False):
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
