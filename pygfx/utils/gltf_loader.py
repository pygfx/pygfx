"""
Utilities to load gltf/glb files.

References: 
https://raw.githubusercontent.com/KhronosGroup/glTF/main/specification/2.0/figures/gltfOverview-2.0.0d.png
https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

Experimental, not yet fully implemented.
"""

import pygfx as gfx
import numpy as np
import pylinalg as la

from importlib.util import find_spec

from functools import cache


accessor_type_size_map = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

component_type_map = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

gltf_attr_map = {
    "POSITION": "positions",
    "NORMAL": "normals",
    "TANGENT": "tangents",
    "TEXCOORD_0": "texcoords",
    "TEXCOORD_1": "texcoords1",
    "COLOR_0": "colors",
    "JOINTS_0": "skin_indices",
    "WEIGHTS_0": "skin_weights",  
}


class GLTF:
    def __init__(self, path):
        self._path = path
        self.scene = None
        self.scenes = []
        self.cameras = []
        self.animations = []

        # self._nodes_cache = {}
        # self._accessors_cache = {}

        self.__inner_load()

    @classmethod
    def load(cls, path):
        '''Load the whole gltf file, including meshes, skeletons, cameras, and animations.'''

        gltf = GLTF(path)

        # TODO: 
        # gltf.cameras
        # gltf.animations
        return gltf
    
    @classmethod
    def load_mesh(cls, path, materials=True):
        '''Only load meshes from a gltf file, without skeletons, and no transformations applied.'''

        gltf = GLTF(path)
        meshes = []
        for gltf_mesh in gltf._gltf.model.meshes:
            mesh = gltf._load_gltf_mesh_by_info(gltf_mesh, load_material=materials)
            meshes.extend(mesh)
        return meshes

    def __inner_load(self):
        import gltflib # noqa
        path = self._path
        self._gltf = gltflib.GLTF.load(path, load_file_resources=True)

        extensions_required = self._gltf.model.extensionsRequired

        if extensions_required is not None:
            gfx.utils.logger.warning(f"This GLTF required extensions: {extensions_required}, which are not supported yet.")
            # rasise or ignore?
            # raise NotImplementedError(f"This GLTF required extensions: {extensions_required}, which are not supported yet.")

        extensions_used = self._gltf.model.extensionsUsed
        if extensions_used is not None:
            gfx.utils.logger.warning(f"This GLTF used extensions: {extensions_used}, which are not supported yet, so the display may not be so correct.")

        # bind the actual data to the buffers
        for buffer in self._gltf.model.buffers:
            buffer.data = self._get_resource_by_uri(buffer.uri).data

        # mark the node types
        self._node_marks = self._mark_nodes()
        self.scenes = self._load_scenes()
        if self._gltf.model.scene is not None:
            self.scene = self.scenes[self._gltf.model.scene]
        return self
    
    @cache
    def _get_resource_by_uri(self, uri):
        for resource in self._gltf.resources:
            if resource.uri == uri:
                return resource
        ValueError(f"Buffer data not found for buffer {uri}")


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
    
    @cache
    def _load_node(self, node_index):
        gltf = self._gltf
        node_marks = self._node_marks
        # nodes_cache = self._nodes_cache

        # if node_index in nodes_cache:
        #     return nodes_cache[node_index]
        
        node = gltf.model.nodes[node_index]

        if node.matrix is not None:
            matrix = np.array(node.matrix).reshape(4,4).T
        else:
            translation= node.translation or [0, 0, 0]
            rotation = node.rotation or [0, 0, 0, 1]
            scale = node.scale or [1, 1, 1]
            matrix = la.mat_compose(translation, rotation, scale)

        node_mark = node_marks[node_index]

        if node_mark == "Bone":
            node_obj = gfx.Bone()
        elif node_mark == "Camera":
            # TODO: implement camera loading
            node_obj = gfx.PerspectiveCamera(45, 1) 
            self.cameras.append(node_obj)
        elif node.mesh is not None: # Mesh or SkinnedMesh
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

        # nodes_cache[node_index] = node_obj

        if node.children:
            for child in node.children:
                child_obj = self._load_node(child)
                node_obj.add(child_obj)

        return node_obj


    def _load_gltf_mesh_by_info(self, mesh, skin_index=None, load_material=True):
        meshes = []
        for primitive in mesh.primitives:
            geometry = self._load_gltf_geometry(primitive)
            if load_material and primitive.material is not None:
                material = self._load_gltf_material(primitive.material)
            else:
                material = None

            primitive_mode = primitive.mode or 4

            if primitive_mode == 0:
                material = material or gfx.PointsMaterial()
                gfx_mesh = gfx.Points(geometry, material)
            
            elif primitive_mode in (1, 2, 3):
                material = material or gfx.LineSegmentMaterial()
                gfx_mesh = gfx.Line(geometry, material)
            
            elif primitive_mode in (4, 5):
                material = material or gfx.MeshBasicMaterial()
                if skin_index is not None:
                    gfx_mesh = gfx.SkinnedMesh(geometry, material)

                    skeleton = self._load_skins(skin_index)

                    gfx_mesh.bind(skeleton, np.identity(4))
                else:
                    gfx_mesh = gfx.Mesh(geometry, material)
            else:
                raise ValueError(f"Unsupported primitive mode: {primitive.mode}")
            
            meshes.append(gfx_mesh)

        return meshes
    
    @cache
    def _load_gltf_mesh(self, mesh_index, skin_index=None, load_material=True):
        mesh = self._gltf.model.meshes[mesh_index]
        return self._load_gltf_mesh_by_info(mesh, skin_index, load_material)

    @cache
    def _load_gltf_material(self, material_index):
        material = self._gltf.model.materials[material_index]
        pbrMetallicRoughness = material.pbrMetallicRoughness

        if pbrMetallicRoughness is not None:
            gfx_material = gfx.MeshStandardMaterial()

            if pbrMetallicRoughness.baseColorTexture is not None:
                gfx_material.map = self._load_gltf_texture(pbrMetallicRoughness.baseColorTexture)

            if pbrMetallicRoughness.metallicRoughnessTexture is not None:
                metallic_roughness_map = self._load_gltf_texture(
                    pbrMetallicRoughness.metallicRoughnessTexture
                )
                gfx_material.roughness_map = metallic_roughness_map
                gfx_material.metalness_map = metallic_roughness_map
                gfx_material.roughness = 1.0
                gfx_material.metalness = 1.0

            if pbrMetallicRoughness.roughnessFactor is not None:
                gfx_material.roughness = pbrMetallicRoughness.roughnessFactor

            if pbrMetallicRoughness.metallicFactor is not None:
                gfx_material.metalness = pbrMetallicRoughness.metallicFactor
        
        else:
            gfx_material = gfx.MeshBasicMaterial()

        if material.normalTexture is not None:
            gfx_material.normal_map = self._load_gltf_texture(material.normalTexture)
            scale_factor = material.normalTexture.scale
            if scale_factor is None:
                scale_factor = 1.0
            
            # See: https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0/NormalTangentTest#problem-flipped-y-axis-or-flipped-green-channel
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

    @cache
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
    
    @cache
    def _load_gltf_sampler(self, sampler_index):
        sampler = self._gltf.model.samplers[sampler_index]
        # print(sampler)
        # Sampler( magFilter=9729, minFilter=9987, wrapS=None, wrapT=None)
        return sampler

    @cache
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

        # if image_info.mimeType == "image/png":
        #     image = iio.imread(image_data, extension=".png")
        # elif image_info.mimeType == "image/jpeg":
        #     image = iio.imread(image_data, extension=".jpg")
        # else:
        #     raise ValueError(f"Unsupported image type: {image_info.mimeType}")
        return image

    def _load_gltf_geometry(self, primitive):
        indices_accessor = primitive.indices
        indices = self._load_accessor(indices_accessor).reshape(-1, 3)

        geometry_args = {"indices": indices}

        for attr, accessor_index in primitive.attributes.__dict__.items():
            if accessor_index is not None:
                geometry_attr = gltf_attr_map[attr]
                geometry_args[geometry_attr] = self._load_accessor(accessor_index)

        return gfx.Geometry(**geometry_args)
    
    @cache
    def _load_skins(self, skin_index):
        skin = self._gltf.model.skins[skin_index]
        bones = [self._load_node(index) for index in skin.joints]
        inverse_bind_matrices = self._load_accessor(skin.inverseBindMatrices)

        bone_inverses = []
        for matrices in inverse_bind_matrices:
            bone_inverse = np.array(matrices).reshape(4,4).T
            bone_inverses.append(bone_inverse)

        skeleton = gfx.Skeleton(bones, bone_inverses)
        return skeleton
    
    @cache
    def _get_buffer_memory_view(self, buffer_view_index):
        gltf = self._gltf
        buffer_view = gltf.model.bufferViews[buffer_view_index]
        buffer = gltf.model.buffers[buffer_view.buffer]
        m = memoryview(buffer.data)
        view = m[buffer_view.byteOffset: buffer_view.byteOffset + buffer_view.byteLength]
        return view
    
    @cache
    def _load_accessor(self, accessor_index):
        # if accessor_index in self._accessors_cache:
        #     return self._accessors_cache[accessor_index]

        gltf = self._gltf
        accessor = gltf.model.accessors[accessor_index]

        buffer_view = gltf.model.bufferViews[accessor.bufferView]
        view = self._get_buffer_memory_view(accessor.bufferView)
        
        if buffer_view.byteStride is not None:
            # Interleaved buffer, implement this later
            raise NotImplementedError
        else:
            accessor_type = accessor.type
            accessor_component_type = accessor.componentType
            accessor_count = accessor.count
            dtype = component_type_map[accessor_component_type]
            accessor_offset = accessor.byteOffset or 0

            accessor_type_size = accessor_type_size_map[accessor_type]
            ar = np.frombuffer(view, dtype=dtype, offset=accessor_offset, count=accessor_count * accessor_type_size)
            if accessor_type_size > 1:
                ar = ar.reshape(accessor_count, accessor_type_size)

            # pygfx not support int8, int16, uint8, uint16 now
            if ar.dtype == np.uint8 or ar.dtype == np.uint16:
                ar = ar.astype(np.uint32)
            if ar.dtype == np.int8 or ar.dtype == np.int16:
                ar = ar.astype(np.int32)

            # self._accessors_cache[accessor_index] = ar
            return ar






def print_tree(obj: gfx.WorldObject, show_pos=False, show_rot=False, show_scale=False, level = 0):
    name = '- ' * level + f"{obj.__class__.__name__}[{obj.name}]"
    if show_pos:
        name += f"\n{'  ' * level}|- pos: {obj.local.position}"
    if show_rot:
        name += f"\n{'  ' * level}|- rot: {obj.local.rotation}"
    if show_scale:
        name += f"\n{'  ' * level}|- scale: {obj.local.scale}"

    print(name)

    for child in obj.children:
        print_tree(child, show_pos=show_pos, show_rot=show_rot, show_scale=show_scale, level = level+1)


