import math
import numpy as np
import wgpu  # only for flags/enums


from ....objects import Mesh, InstancedMesh, SkinnedMesh, WorldObject, Line, Points
from ....resources import Buffer, Texture, TextureMap
from ....utils import normals_from_vertices
from ....materials import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshToonMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
    MeshStandardMaterial,
    MeshPhysicalMaterial,
)

from .. import (
    register_wgpu_render_function,
    BaseShader,
    Binding,
    RenderMask,
    nchannels_from_format,
    to_texture_format,
    GfxSampler,
    GfxTextureView,
    load_wgsl,
)


@register_wgpu_render_function(WorldObject, MeshBasicMaterial)
class MeshShader(BaseShader):
    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

        material = wobject.material
        geometry = wobject.geometry

        # Is this an instanced mesh?
        self["instanced"] = isinstance(wobject, InstancedMesh)

        # Is this a skinned mesh?
        self["use_skinning"] = isinstance(wobject, SkinnedMesh)

        # Is this a morphing mesh?
        morph_attrs = [
            getattr(geometry, name, None)
            for name in ["morph_positions", "morph_normals", "morph_colors"]
        ]
        morph_attrs = [x for x in morph_attrs if x is not None]
        self["use_morph_targets"] = bool(morph_attrs)

        # Is this a wireframe mesh?
        self["wireframe"] = getattr(material, "wireframe", False)
        self["flat_shading"] = getattr(material, "flat_shading", False)

        # Lighting off in the base class
        self["lighting"] = ""
        self["receive_shadow"] = wobject.receive_shadow

        # Per-vertex color, colormap, or a plane color?
        self["colorspace"] = "srgb"

        color_mode = str(material.color_mode).split(".")[-1]

        self["color_mode"] = color_mode

        if material.map is not None and color_mode in (
            "vertex_map",
            "face_map",
            "auto",
        ):
            self["use_map"] = True
            self["colorspace"] = material.map.texture.colorspace
        else:
            self["use_map"] = False

        if getattr(geometry, "colors", None) and (
            color_mode in ("vertex", "face", "auto")
        ):
            nchannels = nchannels_from_format(geometry.colors.format)
            self["use_vertex_color"] = True
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        else:
            self["color_buffer_channels"] = 0
            self["use_vertex_color"] = False

    def _define_texture_map(self, geometry, map, name, view_dim="2d", check=True):
        if check:
            # Check that the texture is compatible with the texcoord
            self._check_texture(map, geometry, view_dim)
        view = GfxTextureView(map.texture, view_dim=view_dim)

        filter_mode = f"{map.mag_filter}, {map.min_filter}, {map.mipmap_filter}"
        address_mode = f"{map.wrap_s}, {map.wrap_t}"
        sampler = GfxSampler(filter_mode, address_mode)

        self[f"{name}_uv"] = map.uv_channel

        bindings = [
            Binding(f"s_{name}", "sampler/filtering", sampler, "FRAGMENT"),
            Binding(f"t_{name}", "texture/auto", view, "FRAGMENT"),
        ]

        if map.uv_channel not in self["used_uv"]:
            texcoords = getattr(geometry, f"texcoords{map.uv_channel or ''}", None)
            if texcoords is not None:
                bindings.append(
                    Binding(
                        f"s_texcoords{map.uv_channel or ''}",
                        "buffer/read_only_storage",
                        texcoords,
                        "VERTEX",
                    )
                )
                self["used_uv"][map.uv_channel] = texcoords.data.ndim

        return bindings

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        # record all uv channels used by texture maps
        self["used_uv"] = {}  # {uv_channel: texcoords_dim}

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        # Triangles or quads?
        if geometry.indices.data is not None and geometry.indices.data.shape[-1] == 4:
            self["indexer"] = 6
        else:
            self["indexer"] = 3

        # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
        if getattr(geometry, "normals", None) is not None:
            normal_buffer = geometry.normals
        else:
            if geometry.indices.data.shape[-1] == 4:
                n1 = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data[..., :3]
                )
                n2 = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data[..., [0, 2, 3]]
                )
                normal_data = (n1 + n2) / 2
            else:
                normal_data = normals_from_vertices(
                    geometry.positions.data, geometry.indices.data
                )
            normal_buffer = Buffer(normal_data)

        # Init bindings
        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("s_indices", rbuffer, geometry.indices, "VERTEX"),
            Binding("s_positions", rbuffer, geometry.positions, "VERTEX"),
            Binding("s_normals", rbuffer, normal_buffer, "VERTEX"),
        ]

        if self["use_vertex_color"]:
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))

        if getattr(geometry, "tangents", None):
            bindings.append(Binding("s_tangents", rbuffer, geometry.tangents, "VERTEX"))
            self["use_tangent"] = True

        if self["use_map"]:
            map = material.map
            map_fmt, map_dim = to_texture_format(map.texture.format), map.texture.dim
            is_standard_map = (
                map_dim == 2
                and ("norm" in map_fmt or "float" in map_fmt)
                and ("rgb" in map_fmt)  # note: we assume r*, rg* maps are colormap
            )

            if not is_standard_map:
                # It's a 'generic' colormap
                self["use_colormap"] = True
                bindings.extend(
                    self.define_generic_colormap(material.map, geometry.texcoords)
                )
                if 0 not in self["used_uv"]:
                    texcoords = getattr(geometry, "texcoords", None)
                    bindings.append(
                        Binding("s_texcoords", rbuffer, texcoords, "VERTEX")
                    )
                    if texcoords.data.ndim == 1:
                        self["used_uv"][0] = 1
                    else:
                        self["used_uv"][0] = texcoords.data.shape[-1]
            else:
                # It's a classic mesh map
                bindings.extend(self._define_texture_map(geometry, material.map, "map"))

            self["colorspace"] = material.map.texture.colorspace

        if self["use_skinning"]:
            # Skinning requires skin_index and skin_weight buffers
            assert hasattr(geometry, "skin_indices") and hasattr(
                geometry, "skin_weights"
            )

            bindings.append(
                Binding("s_skin_indices", rbuffer, geometry.skin_indices, "VERTEX")
            )
            bindings.append(
                Binding("s_skin_weights", rbuffer, geometry.skin_weights, "VERTEX")
            )

            # Skinning requires a bone_matrices buffer

            skeleton = wobject.skeleton

            bindings.append(
                Binding(
                    "u_bone_matrices",
                    "buffer/uniform",
                    skeleton.bone_matrices_buffer,
                    "VERTEX",
                )
            )

        if self["use_morph_targets"]:
            # Get or create the morph texture and associated info
            try:
                morph_texture_info = geometry["_gfx_morph_texture"]
            except KeyError:
                morph_texture_info = self._encode_morph_texture(geometry, shared)
                geometry["_gfx_morph_texture"] = morph_texture_info
            morph_texture, stride, width, morph_count = morph_texture_info

            morph_target_influences = wobject._morph_target_influences  # buffer

            if morph_texture and morph_target_influences:
                view = GfxTextureView(morph_texture, view_dim="2d-array")
                bindings.append(
                    Binding("t_morph_targets", "texture/auto", view, "VERTEX")
                )

                self["influences_buffer_size"] = morph_target_influences.nitems
                self["morph_targets_count"] = morph_count
                self["morph_targets_stride"] = stride
                self["morph_targets_texture_width"] = width

            else:
                self["use_morph_targets"] = False

        # specular map configs, only for basic and phong materials now
        if getattr(material, "specular_map", None):
            bindings.extend(
                self._define_texture_map(
                    geometry, material.specular_map, "specular_map"
                )
            )
            self["use_specular_map"] = True

        # set emissive_map configs, for phong and standard materials
        if getattr(material, "emissive_map", None):
            bindings.extend(
                self._define_texture_map(
                    geometry, material.emissive_map, "emissive_map"
                )
            )
            self["use_emissive_map"] = True

        # set normal_map configs
        if getattr(material, "normal_map", None):
            bindings.extend(
                self._define_texture_map(geometry, material.normal_map, "normal_map")
            )
            self["use_normal_map"] = True

        # set envmap configs
        if getattr(material, "env_map", None):
            # TODO: Support envmap not only cube, but also equirect (hdr format)

            # special check for env_map
            assert isinstance(material.env_map, TextureMap)
            assert material.env_map.texture.size[2] == 6, "env_map must be a cube map"
            fmt = to_texture_format(material.env_map.texture.format)
            assert "norm" in fmt or "float" in fmt

            bindings.extend(
                self._define_texture_map(
                    geometry, material.env_map, "env_map", view_dim="cube", check=False
                )  # check=False because we don't need texcoords for env_map
            )

            if isinstance(material, MeshStandardMaterial):
                self["USE_IBL"] = True
            elif isinstance(material, MeshBasicMaterial):
                self["use_env_map"] = True
                self["env_combine_mode"] = getattr(
                    material, "env_combine_mode", "MULTIPLY"
                )

            self["env_mapping_mode"] = getattr(
                material, "env_mapping_mode", "CUBE-REFLECTION"
            )

        # set lightmap configs
        if getattr(material, "light_map", None):
            bindings.extend(
                self._define_texture_map(geometry, material.light_map, "light_map")
            )
            self["use_light_map"] = True

        # set aomap configs
        if getattr(material, "ao_map", None):
            bindings.extend(
                self._define_texture_map(geometry, material.ao_map, "ao_map")
            )
            self["use_ao_map"] = True

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(0, bindings)

        # Instanced meshes have an extra storage buffer that we add manually
        bindings1 = {}  # non-auto-generated bindings
        if self["instanced"]:
            bindings1[0] = Binding(
                "s_instance_infos", rbuffer, wobject.instance_buffer, "VERTEX"
            )

        if self["use_morph_targets"]:
            bindings1[1] = Binding(
                "u_morph_target_influences",
                "buffer/uniform",
                morph_target_influences,
                "VERTEX",
            )

        return {
            0: bindings,
            1: bindings1,
        }

    def _encode_morph_texture(self, geometry, shared):
        morph_positions = getattr(geometry, "morph_positions", None)
        morph_normals = getattr(geometry, "morph_normals", None)
        morph_colors = getattr(geometry, "morph_colors", None)

        morph_attrs = [morph_positions, morph_normals, morph_colors]
        morph_attrs = [x for x in morph_attrs if x is not None]
        morph_count = min(len(x) for x in morph_attrs)

        vetex_data_count = 0

        if morph_positions:
            vetex_data_count = 1

        if morph_normals:
            vetex_data_count = 2

        if morph_colors:
            vetex_data_count = 3

        if vetex_data_count == 0:
            return None, None, None, None

        vertice_count = geometry.positions.nitems
        total_count = vertice_count * vetex_data_count
        width = total_count
        height = 1

        max_texture_width = shared.device.limits["max-texture-dimension-2d"]

        if width > max_texture_width:
            height = math.ceil(width / max_texture_width)
            width = max_texture_width

        buffer = np.zeros((morph_count, height * width, 4), dtype=np.float32)

        for i in range(morph_count):
            if morph_positions:
                morph_position = morph_positions[i]
                assert len(morph_position) == vertice_count, (
                    f"Morph target {i} has {len(morph_position)} vertices, expected {vertice_count}"
                )
                morph_position = np.pad(morph_position, ((0, 0), (0, 1)), "constant")
            else:
                morph_position = np.zeros((vertice_count, 4), dtype=np.float32)

            if morph_normals:
                morph_normal = morph_normals[i]
                assert len(morph_normal) == vertice_count, (
                    f"Morph normal {i} has {len(morph_normal)} vertices, expected {vertice_count}"
                )
                morph_normal = np.pad(morph_normal, ((0, 0), (0, 1)), "constant")
            else:
                morph_normal = np.zeros((vertice_count, 4), dtype=np.float32)

            if morph_colors:
                morph_color = morph_colors[i]
                assert len(morph_color) == vertice_count, (
                    f"Morph color {i} has {len(morph_colors[i])} vertices, expected {vertice_count}"
                )
            else:
                morph_color = np.zeros((vertice_count, 4), dtype=np.float32)

            morph_data = np.stack(
                (morph_position, morph_normal, morph_color)[:vetex_data_count], axis=1
            ).reshape(-1, 4)

            buffer[i, :total_count, :] = morph_data

        return (
            Texture(buffer, dim=2, size=(width, height, morph_count)),
            vetex_data_count,
            width,
            morph_count,
        )

    def get_pipeline_info(self, wobject, shared):
        material = wobject.material

        # The MeshMaterial can be applied to lines and points, so that we can fully support gltf
        if isinstance(wobject, Line):
            topology = wgpu.PrimitiveTopology.line_strip
            # todo: wgpu.PrimitiveTopology.line_list, wgpu.PrimitiveTopology.line_loop
        elif isinstance(wobject, Points):
            topology = wgpu.PrimitiveTopology.point_list
        elif isinstance(wobject, Mesh):
            topology = wgpu.PrimitiveTopology.triangle_list
            # todo: wgpu.PrimitiveTopology.triangle_strip
        else:
            raise TypeError(
                f"MeshMaterial cannot be applied to a {wobject.__class.__name__}"
            )

        if material.side == "front":
            cull_mode = wgpu.CullMode.back
        elif material.side == "back":
            cull_mode = wgpu.CullMode.front
        else:  # material.side == "both"
            cull_mode = wgpu.CullMode.none

        return {
            "primitive_topology": topology,
            "cull_mode": cull_mode,
        }

    def get_render_info(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

        if geometry.indices.data is not None and geometry.indices.data.shape[-1] == 4:
            offset, size = geometry.indices.draw_range
            offset, size = 6 * offset, 6 * size
        else:
            offset, size = geometry.indices.draw_range
            offset, size = 3 * offset, 3 * size

        n_instances = 1
        if self["instanced"]:
            n_instances = wobject.instance_buffer.nitems

        render_mask = wobject.render_mask
        if not render_mask:
            render_mask = RenderMask.all
            if material.is_transparent:
                render_mask = RenderMask.transparent
            elif self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask = RenderMask.transparent
                else:
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (1, 3):
                    render_mask = RenderMask.opaque
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (1, 3):
                    render_mask = RenderMask.opaque
            elif self["color_mode"] == "normal":
                render_mask = RenderMask.opaque
            elif self["color_mode"] == "auto":
                render_mask = RenderMask.all
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, n_instances, offset, 0),
            "render_mask": render_mask,
        }

    def _check_texture(self, t, geometry, view_dim):
        assert isinstance(t, TextureMap)
        uv_channel = t.uv_channel
        if uv_channel > 0:
            texcoords = getattr(geometry, f"texcoords{uv_channel}", None)
        else:
            texcoords = getattr(geometry, "texcoords", None)
        assert texcoords is not None, (
            f"Texture {t} requires geometry.texcoords{uv_channel or ''}"
        )

        if view_dim == "1d":
            assert texcoords.data.ndim == 1 or texcoords.data.shape[-1] == 1, (
                f"Texture {t} requires 1D texcoords"
            )
        elif view_dim == "2d" or view_dim == "2d-array":
            assert texcoords.data.ndim == 2 and texcoords.data.shape[-1] == 2, (
                f"Texture {t} requires 2D texcoords"
            )
        elif view_dim == "cube" or view_dim == "3d" or view_dim == "cube-array":
            assert texcoords.data.ndim == 2 and texcoords.data.shape[-1] == 3, (
                f"Texture {t} requires 3D texcoords"
            )
        else:
            raise ValueError(f"Unknown view_dim: {view_dim}")

        fmt = to_texture_format(t.texture.format)
        assert "norm" in fmt or "float" in fmt

    def get_code(self):
        return load_wgsl("mesh.wgsl")


@register_wgpu_render_function(Mesh, MeshNormalMaterial)
class MeshNormalShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["color_mode"] = "normal"
        self["colormap_dim"] = ""  # disable texture if there happens to be one


@register_wgpu_render_function(Mesh, MeshPhongMaterial)
class MeshPhongShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "phong"


@register_wgpu_render_function(Mesh, MeshToonMaterial)
class MeshToonShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "toon"

    def get_bindings(self, wobject, shared):
        result = super().get_bindings(wobject, shared)

        geometry = wobject.geometry
        material = wobject.material

        bindings = []

        if material.gradient_map is not None:
            bindings.extend(
                self._define_texture_map(
                    geometry, material.gradient_map, "gradient_map"
                )
            )
            self["use_gradient_map"] = True

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(2, bindings)

        # Update result
        result[2] = bindings
        return result


@register_wgpu_render_function(WorldObject, MeshStandardMaterial)
class MeshStandardShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "pbr"

    def get_bindings(self, wobject, shared):
        result = super().get_bindings(wobject, shared)

        geometry = wobject.geometry
        material = wobject.material

        bindings = []

        if material.roughness_map is not None:
            bindings.extend(
                self._define_texture_map(
                    geometry, material.roughness_map, "roughness_map"
                )
            )
            self["use_roughness_map"] = True

        if material.metalness_map is not None:
            bindings.extend(
                self._define_texture_map(
                    geometry, material.metalness_map, "metalness_map"
                )
            )
            self["use_metalness_map"] = True

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(2, bindings)

        # Update result
        result[2] = bindings
        return result


@register_wgpu_render_function(WorldObject, MeshPhysicalMaterial)
class MeshPhysicalShader(MeshStandardShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "pbr"
        self["USE_IOR"] = True
        self["USE_SPECULAR"] = True

    def get_bindings(self, wobject, shared):
        result = super().get_bindings(wobject, shared)

        geometry = wobject.geometry
        material = wobject.material

        bindings = []

        if material.specular_intensity_map is not None:
            bindings.extend(
                self._define_texture_map(
                    geometry, material.specular_intensity_map, "specular_intensity_map"
                )
            )
            self["use_specular_intensity_map"] = True

        # clearcoat
        if material.clearcoat:
            self["USE_CLEARCOAT"] = True

            if material.clearcoat_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.clearcoat_map, "clearcoat_map"
                    )
                )
                self["use_clearcoat_map"] = True

            if material.clearcoat_roughness_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry,
                        material.clearcoat_roughness_map,
                        "clearcoat_roughness_map",
                    )
                )
                self["use_clearcoat_roughness_map"] = True

            if material.clearcoat_normal_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.clearcoat_normal_map, "clearcoat_normal_map"
                    )
                )
                self["use_clearcoat_normal_map"] = True

        # iridescence
        if material.iridescence:
            self["USE_IRIDESCENCE"] = True

            if material.iridescence_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.iridescence_map, "iridescence_map"
                    )
                )
                self["use_iridescence_map"] = True

            if material.iridescence_thickness_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry,
                        material.iridescence_thickness_map,
                        "iridescence_thickness_map",
                    )
                )
                self["use_iridescence_thickness_map"] = True

        # sheen
        if material.sheen:
            self["USE_SHEEN"] = True

            if material.sheen_color_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.sheen_color_map, "sheen_color_map"
                    )
                )
                self["use_sheen_color_map"] = True

            if material.sheen_roughness_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.sheen_roughness_map, "sheen_roughness_map"
                    )
                )
                self["use_sheen_roughness_map"] = True

        # anisotropy
        if material.anisotropy:
            self["USE_ANISOTROPY"] = True

            if material.anisotropy_map is not None:
                bindings.extend(
                    self._define_texture_map(
                        geometry, material.anisotropy_map, "anisotropy_map"
                    )
                )
                self["use_anisotropy_map"] = True

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(3, bindings)

        # Update result
        result[3] = bindings
        return result


@register_wgpu_render_function(Mesh, MeshNormalLinesMaterial)
class MeshNormalLinesShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["color_mode"] = "uniform"
        self["lighting"] = ""
        self["wireframe"] = False

    def get_pipeline_info(self, wobject, shared):
        d = super().get_pipeline_info(wobject, shared)
        d["primitive_topology"] = wgpu.PrimitiveTopology.line_list
        return d

    def get_render_info(self, wobject, shared):
        # We directly look at the vertex data, so geometry.indices.draw_range is ignored.
        d = super().get_render_info(wobject, shared)
        d["indices"] = wobject.geometry.positions.nitems * 2, d["indices"][1]
        return d

    def get_code(self):
        return load_wgsl("mesh_normal_lines.wgsl")


@register_wgpu_render_function(Mesh, MeshSliceMaterial)
class MeshSliceShader(BaseShader):
    """Shader for rendering mesh slices."""

    type = "render"

    def __init__(self, wobject):
        super().__init__(wobject)

        material = wobject.material
        geometry = wobject.geometry

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
            else:
                self["color_mode"] = "uniform"
                self["color_buffer_channels"] = 0
        elif color_mode == "uniform":
            self["color_mode"] = "uniform"
            self["color_buffer_channels"] = 0
        elif color_mode == "vertex":
            nchannels = nchannels_from_format(geometry.colors.format)
            self["color_mode"] = "vertex"
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "face":
            nchannels = nchannels_from_format(geometry.colors.format)
            self["color_mode"] = "face"
            self["color_buffer_channels"] = nchannels
            if nchannels not in (1, 2, 3, 4):
                raise ValueError(f"Geometry.colors needs 1-4 columns, not {nchannels}")
        elif color_mode == "vertex_map":
            self["color_mode"] = "vertex_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError("Cannot apply colormap is no material.map is set.")
        elif color_mode == "face_map":
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError("Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        # It would technically be possible to implement colormapping or
        # per-vertex colors, but its a tricky dance to get the per-vertex
        # data (e.g. texcoords) into a varying. And because the visual
        # result is a line, its likely that in most use-cases a uniform
        # color is preferred anyway. So for now we don't implement that.

        geometry = wobject.geometry
        material = wobject.material

        # Init uniform bindings
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
        ]

        # We're assuming the presence of an index buffer for now
        assert getattr(geometry, "indices", None)

        # Init storage buffer bindings
        rbuffer = "buffer/read_only_storage"
        bindings.append(Binding("s_indices", rbuffer, geometry.indices, "VERTEX"))
        bindings.append(Binding("s_positions", rbuffer, geometry.positions, "VERTEX"))

        # Bindings for color
        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.extend(
                self.define_generic_colormap(material.map, geometry.texcoords)
            )

        # Let the shader generate code for our bindings
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(0, bindings)

        return {
            0: bindings,
        }

    def get_pipeline_info(self, wobject, shared):
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": wgpu.CullMode.none,
        }

    def get_render_info(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        offset, size = geometry.indices.draw_range
        offset, size = offset * 6, size * 6

        render_mask = 0
        if wobject.render_mask:
            render_mask = wobject.render_mask
        elif material.is_transparent:
            render_mask = RenderMask.transparent
        else:
            # Get what passes are needed for the color
            if self["color_mode"] == "uniform":
                if material.color_is_transparent:
                    render_mask |= RenderMask.transparent
                else:
                    render_mask |= RenderMask.opaque
            elif self["color_mode"] in ("vertex", "face"):
                if self["color_buffer_channels"] in (1, 3):
                    render_mask |= RenderMask.opaque
                else:
                    render_mask |= RenderMask.all
            elif self["color_mode"] in ("vertex_map", "face_map"):
                if self["colormap_nchannels"] in (1, 3):
                    render_mask |= RenderMask.opaque
                else:
                    render_mask |= RenderMask.all
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, 1, offset, 0),
            "render_mask": render_mask,
        }

    def get_code(self):
        return load_wgsl("mesh_slice.wgsl")
