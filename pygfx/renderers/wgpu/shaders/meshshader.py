import math
import numpy as np
import wgpu  # only for flags/enums


from ....objects import Mesh, InstancedMesh, SkinnedMesh
from ....resources import Buffer, Texture
from ....utils import normals_from_vertices
from ....materials import (
    MeshBasicMaterial,
    MeshPhongMaterial,
    MeshToonMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshSliceMaterial,
    MeshStandardMaterial,
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


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
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
        self["use_morph_targets"] = (
            getattr(geometry, "morph_positions", None)
            or getattr(geometry, "morph_normals", None)
            or getattr(geometry, "morph_colors", None)
        )

        # Is this a wireframe mesh?
        self["wireframe"] = getattr(material, "wireframe", False)
        self["flat_shading"] = getattr(material, "flat_shading", False)

        # Lighting off in the base class
        self["lighting"] = ""
        self["receive_shadow"] = wobject.receive_shadow

        # Per-vertex color, colormap, or a plane color?
        self["colorspace"] = "srgb"

        color_mode = str(material.color_mode).split(".")[-1]
        if color_mode == "auto":
            if material.map is not None:
                self["color_mode"] = "vertex_map"
                self["color_buffer_channels"] = 0
                self["colorspace"] = material.map.colorspace
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
            self["colorspace"] = material.map.colorspace
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        elif color_mode == "face_map":
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
            self["colorspace"] = material.map.colorspace
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        else:
            raise RuntimeError(f"Unknown color_mode: '{color_mode}'")

    def get_bindings(self, wobject, shared):
        geometry = wobject.geometry
        material = wobject.material

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

        # We always need texcoords, not only for colormap
        if hasattr(geometry, "texcoords") and geometry.texcoords is not None:
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )

        if hasattr(geometry, "texcoords1") and geometry.texcoords1 is not None:
            bindings.append(
                Binding("s_texcoords1", rbuffer, geometry.texcoords1, "VERTEX")
            )
            self["use_texcoords1"] = True

        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.extend(
                # todo: check uv is present and has the right shape. (introduce uv channel for texture)
                # todo: unify the logic with other maps, introduce uv channel and texture sampler
                self.define_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
            )

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
            morph_texture, stride, width, morph_count = getattr(
                geometry, "_gfx_morph_texture", (None, None, None, None)
            )
            if morph_texture is None:
                morph_texture, stride, width, morph_count = self._encode_morph_texture(
                    geometry
                )
                geometry._gfx_morph_texture = (
                    morph_texture,
                    stride,
                    width,
                    morph_count,
                )

            morph_target_influences = (
                wobject._morph_target_influences
            )  # the influences buffer

            if morph_texture and morph_target_influences:
                view = GfxTextureView(morph_texture, view_dim="2d-array")
                bindings.append(
                    Binding("t_morph_targets", "texture/auto", view, "VERTEX")
                )

                self["morph_targets_count"] = min(
                    morph_target_influences.nitems, morph_count
                )
                self["morph_targets_stride"] = stride
                self["morph_targets_texture_width"] = width

            else:
                self["use_morph_targets"] = False

        F = "FRAGMENT"  # noqa: N806
        sampling = "sampler/filtering"
        sampler = GfxSampler(material.map_interpolation, "repeat")
        texturing = "texture/auto"

        # todo, check uv is present and has the right shape. (introduce uv channel for texture)

        # specular map configs, only for basic and phong materials now
        if getattr(material, "specular_map", None) and not isinstance(
            material, MeshStandardMaterial
        ):
            self._check_texture(material.specular_map)
            view = GfxTextureView(material.specular_map, view_dim="2d")
            self["use_specular_map"] = True
            bindings.append(Binding("s_specular_map", sampling, sampler, F))
            bindings.append(Binding("t_specular_map", texturing, view, F))

        # set emissive_map configs, for phong and standard materials
        if getattr(material, "emissive_map", None):
            self._check_texture(material.emissive_map)
            view = GfxTextureView(material.emissive_map, view_dim="2d")
            self["use_emissive_map"] = True
            bindings.append(Binding("s_emissive_map", sampling, sampler, F))
            bindings.append(Binding("t_emissive_map", texturing, view, F))

        # set normal_map configs
        if getattr(material, "normal_map", None):
            self._check_texture(material.normal_map)
            view = GfxTextureView(material.normal_map, view_dim="2d")
            self["use_normal_map"] = True
            bindings.append(Binding("s_normal_map", sampling, sampler, F))
            bindings.append(Binding("t_normal_map", texturing, view, F))

        # set envmap configs
        if getattr(material, "env_map", None):
            self._check_texture(material.env_map, 6)
            # TODO: Support envmap not only cube, but also equirect (hdr format)
            view = GfxTextureView(material.env_map, view_dim="cube")
            bindings.append(Binding("s_env_map", sampling, sampler, F))
            bindings.append(Binding("t_env_map", texturing, view, F))

            if isinstance(material, MeshStandardMaterial):
                self["use_IBL"] = True
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
            if "use_texcoords1" not in self.kwargs:
                raise ValueError(
                    "Light map requires a second set of texture coordinates (geometry.texcoords1), but it is not present."
                )
            else:
                self._check_texture(material.light_map)
                view = GfxTextureView(material.light_map, view_dim="2d")
                self["use_light_map"] = True
                bindings.append(Binding("s_light_map", sampling, sampler, F))
                bindings.append(Binding("t_light_map", texturing, view, F))

        # set aomap configs
        if getattr(material, "ao_map", None):
            if "use_texcoords1" not in self.kwargs:
                raise ValueError(
                    "AoMap requires a second set of texture coordinates (geometry.texcoords1), but it is not present."
                )
            else:
                self._check_texture(material.ao_map)
                view = GfxTextureView(material.ao_map, view_dim="2d")
                self["use_ao_map"] = True
                bindings.append(Binding("s_ao_map", sampling, sampler, F))
                bindings.append(Binding("t_ao_map", texturing, view, F))

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

    def _encode_morph_texture(self, geometry):
        morph_positions = getattr(geometry, "morph_positions", None)
        morph_normals = getattr(geometry, "morph_normals", None)
        morph_colors = getattr(geometry, "morph_colors", None)

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

        max_texture_width = 4096  # TODO: use wgpu capabilities "max_texture_size"

        if width > max_texture_width:
            height = math.ceil(width / max_texture_width)
            width = max_texture_width

        morph_count = len(morph_positions or morph_normals or morph_colors or [])

        buffer = np.zeros((morph_count, height * width, 4), dtype=np.float32)

        for i in range(morph_count):
            if morph_positions:
                morph_position = morph_positions[i]
                assert (
                    len(morph_position) == vertice_count
                ), f"Morph target {i} has {len(morph_position)} vertices, expected {vertice_count}"
                morph_position = np.pad(morph_position, ((0, 0), (0, 1)), "constant")
            else:
                morph_position = np.zeros((vertice_count, 4), dtype=np.float32)

            if morph_normals:
                morph_normal = morph_normals[i]
                assert (
                    len(morph_normal) == vertice_count
                ), f"Morph normal {i} has {len(morph_normal)} vertices, expected {vertice_count}"
                morph_normal = np.pad(morph_normal, ((0, 0), (0, 1)), "constant")
            else:
                morph_normal = np.zeros((vertice_count, 4), dtype=np.float32)

            if morph_colors:
                morph_color = morph_colors[i]
                assert (
                    len(morph_color) == vertice_count
                ), f"Morph color {i} has {len(morph_colors[i])} vertices, expected {vertice_count}"
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

        topology = wgpu.PrimitiveTopology.triangle_list

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
            else:
                raise RuntimeError(f"Unexpected color mode {self['color_mode']}")

        return {
            "indices": (size, n_instances, offset, 0),
            "render_mask": render_mask,
        }

    def _check_texture(self, t, size2=1):
        assert isinstance(t, Texture)
        assert t.size[2] == size2
        fmt = to_texture_format(t.format)
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

        F = "FRAGMENT"  # noqa: N806

        # We need uv to use the maps, so if uv not exist, ignore all maps
        if hasattr(geometry, "texcoords") and geometry.texcoords is not None:
            # Texcoords must always be nx2 since it used for all texture maps.
            nchannels = nchannels_from_format(geometry.texcoords.format)
            if not (geometry.texcoords.data.ndim == 2 and nchannels == 2):
                raise ValueError("For toon material, the texcoords must be Nx2")

            if material.gradient_map is not None:
                self._check_texture(material.gradient_map)
                view = GfxTextureView(material.gradient_map, view_dim="2d")
                self["use_gradient_map"] = True
                bindings.append(
                    Binding(
                        "s_gradient_map",
                        "sampler/filtering",
                        GfxSampler("nearest", "repeat"),
                        F,
                    )
                )  # gradient map is always nearest sampling
                bindings.append(Binding("t_gradient_map", "texture/auto", view, F))

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(2, bindings)

        # Update result
        result[2] = bindings
        return result


@register_wgpu_render_function(Mesh, MeshStandardMaterial)
class MeshStandardShader(MeshShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["lighting"] = "pbr"

    def get_bindings(self, wobject, shared):
        result = super().get_bindings(wobject, shared)

        geometry = wobject.geometry
        material = wobject.material

        bindings = []

        F = "FRAGMENT"  # noqa: N806
        sampling = "sampler/filtering"
        sampler = GfxSampler(material.map_interpolation, "repeat")
        texturing = "texture/auto"

        # We need uv to use the maps, so if uv not exist, ignore all maps
        if hasattr(geometry, "texcoords") and geometry.texcoords is not None:
            # Texcoords must always be nx2 since it used for all texture maps.
            nchannels = nchannels_from_format(geometry.texcoords.format)
            if not (geometry.texcoords.data.ndim == 2 and nchannels == 2):
                raise ValueError("For standard material, the texcoords must be Nx2")

            if material.roughness_map is not None:
                self._check_texture(material.roughness_map)
                view = GfxTextureView(material.roughness_map, view_dim="2d")
                self["use_roughness_map"] = True
                bindings.append(Binding("s_roughness_map", sampling, sampler, F))
                bindings.append(Binding("t_roughness_map", texturing, view, F))

            if material.metalness_map is not None:
                self._check_texture(material.metalness_map)
                view = GfxTextureView(material.metalness_map, view_dim="2d")
                self["use_metalness_map"] = True
                bindings.append(Binding("s_metalness_map", sampling, sampler, F))
                bindings.append(Binding("t_metalness_map", texturing, view, F))

        # Define shader code for binding
        bindings = {i: binding for i, binding in enumerate(bindings)}
        self.define_bindings(2, bindings)

        # Update result
        result[2] = bindings
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
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
        elif color_mode == "face_map":
            self["color_mode"] = "face_map"
            self["color_buffer_channels"] = 0
            if material.map is None:
                raise ValueError(f"Cannot apply colormap is no material.map is set.")
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
                self.define_colormap(
                    material.map, geometry.texcoords, material.map_interpolation
                )
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
        material = wobject.material  # noqa
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
