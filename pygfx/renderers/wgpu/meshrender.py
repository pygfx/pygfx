import wgpu  # only for flags/enums
import pyshader
import numpy as np
from pyshader import python2shader
from pyshader import f32, vec2, vec3, vec4, Array


from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Mesh
from ...materials import (
    MeshBasicMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshPhongMaterial,
)
from ...datawrappers import Buffer, Texture, TextureView


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
def mesh_renderer(wobject, render_info):
    """ Render function capable of rendering meshes.
    """

    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize some pipeline things
    topology = wgpu.PrimitiveTopology.triangle_list
    vertex_shader = vertex_shader_mesh
    fragment_shader = fragment_shader_simple

    # Use index buffer if present on the geometry
    index_buffer = getattr(geometry, "index", None)
    index_buffer = index_buffer if isinstance(index_buffer, Buffer) else None

    if index_buffer:
        n = len(index_buffer.data)
    else:
        n = len(vertex_buffers[0].data)

    # Collect vertex buffers
    vertex_buffers = []
    vertex_buffers.append(geometry.positions)
    if getattr(geometry, "texcoords", None) is not None:
        vertex_buffers.append(geometry.texcoords)

    # todo: fail if no texcoords are given, make vertex_buffers a dict!

    # Normals
    if getattr(geometry, "normals", None) is not None:
        normal_buffer = geometry.normals
    else:
        normal_data = _calculate_normals(geometry.positions.data, index_buffer.data)
        normal_buffer = Buffer(normal_data, usage="vertex|storage")
    vertex_buffers.append(normal_buffer)

    bindings0 = {0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo)}
    bindings1 = {}

    bindings1[0] = wgpu.BindingType.uniform_buffer, material.uniform_buffer

    # Collect texture and sampler
    if isinstance(material, MeshNormalMaterial):
        fragment_shader = fragment_shader_normals
    elif isinstance(material, MeshNormalLinesMaterial):
        topology = wgpu.PrimitiveTopology.line_list
        vertex_shader = vertex_shader_normal_lines
        fragment_shader = fragment_shader_simple
        bindings0[1] = wgpu.BindingType.readonly_storage_buffer, geometry.positions
        bindings0[2] = wgpu.BindingType.readonly_storage_buffer, normal_buffer
        vertex_buffers = []
        index_buffer = None
        n = geometry.positions.nitems * 2
    elif isinstance(material, MeshPhongMaterial):
        fragment_shader = fragment_shader_phong
    elif material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("material.map is present, but geometry has no texcoords")
        bindings1[1] = wgpu.BindingType.sampler, material.map
        bindings1[2] = wgpu.BindingType.sampled_texture, material.map
        if "rgba" in material.map.format:
            fragment_shader = fragment_shader_textured_rgba
        else:
            fragment_shader = fragment_shader_textured_gray
        # Use a version of the shader for float textures if necessary
        if "float" in material.map.format:
            if not hasattr(fragment_shader, "float_version"):
                func = fragment_shader.input
                tex_anno = func.__annotations__["t_tex"]
                func.__annotations__["t_tex"] = tex_anno[:2] + ("2d f32",)
                fragment_shader.float_version = python2shader(func)
                func.__annotations__["t_tex"] = tex_anno
            fragment_shader = fragment_shader.float_version

    # Put it together!

    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


# Taken from Vispy
def _calculate_normals(rr, tris):
    """Efficiently compute vertex normals for triangulated surface"""
    # ensure highest precision for our summation/vectorization "trick"
    rr = rr[:, :3].astype(np.float64)
    tris = tris.reshape(-1, 3)
    # first, compute triangle normals
    r1 = rr[tris[:, 0], :]
    r2 = rr[tris[:, 1], :]
    r3 = rr[tris[:, 2], :]
    tri_nn = np.cross((r2 - r1), (r3 - r1))

    # Triangle normals and areas
    size = np.sqrt(np.sum(tri_nn * tri_nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    tri_nn /= size[:, np.newaxis]

    npts = len(rr)

    # the following code replaces this, but is faster (vectorized):
    #
    # for p, verts in enumerate(tris):
    #     nn[verts, :] += tri_nn[p, :]
    #
    nn = np.zeros((npts, 3))
    for verts in tris.T:  # note this only loops 3x (number of verts per tri)
        for idx in range(3):  # x, y, z
            nn[:, idx] += np.bincount(
                verts.astype(np.int32), tri_nn[:, idx], minlength=npts
            )
    size = np.sqrt(np.sum(nn * nn, axis=1))
    size[size == 0] = 1.0  # prevent ugly divide-by-zero
    nn /= size[:, np.newaxis]
    return nn.astype(np.float32)


# %% Shaders


@python2shader
def vertex_shader_mesh(
    in_pos: (pyshader.RES_INPUT, 0, vec4),
    in_texcoord: (pyshader.RES_INPUT, 1, vec2),
    in_normal: (pyshader.RES_INPUT, 2, vec3),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
    v_normal: (pyshader.RES_OUTPUT, 1, vec3),
    v_view: (pyshader.RES_OUTPUT, 2, vec3),
    v_light: (pyshader.RES_OUTPUT, 3, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
):
    world_pos = u_stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    ndc_to_world = matrix_inverse(
        u_stdinfo.cam_transform * u_stdinfo.projection_transform
    )

    normal_vec = u_stdinfo.world_transform * vec4(in_normal.xyz, 1.0)

    view_vec = ndc_to_world * vec4(0, 0, 1, 1)
    view_vec = normalize(view_vec.xyz / view_vec.w)

    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = in_texcoord  # noqa - shader output

    # Vectors for lighting, all in world coordinates
    v_normal = normal_vec  # noqa
    v_view = view_vec  # noqa
    v_light = view_vec  # noqa


@python2shader
def vertex_shader_normal_lines(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_pos: (pyshader.RES_BUFFER, (0, 1), Array(vec4)),
    buf_normal: (pyshader.RES_BUFFER, (0, 2), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
):

    r = index % 2
    i = (index - r) // 2

    pos = buf_pos[i].xyz
    normal = vec3(buf_normal[i * 3 + 0], buf_normal[i * 3 + 1], buf_normal[i * 3 + 2])
    pos = pos + f32(r) * normal * 0.1  # todo: allow user to specify normal length

    world_pos = u_stdinfo.world_transform * vec4(pos, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader output


@python2shader
def fragment_shader_simple(
    u_mesh: (pyshader.RES_UNIFORM, (1, 0), MeshBasicMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    """ Just draw the fragment in the mesh's color.
    """
    out_color = u_mesh.color  # noqa - shader output


@python2shader
def fragment_shader_normals(
    v_normal: (pyshader.RES_INPUT, 1, vec3), out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    """ Draws the mesh in a color derived from the normal.
    """
    v = normalize(v_normal) * 0.5 + 0.5
    out_color = vec4(v, 1.0)  # noqa - shader output


@python2shader
def fragment_shader_textured_gray(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    u_mesh: (pyshader.RES_UNIFORM, (1, 0), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 1), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 2), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    val = f32(t_tex.sample(s_sam, v_texcoord).r)
    val = (val - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = vec4(val, val, val, 1.0)  # noqa - shader output


@python2shader
def fragment_shader_textured_rgba(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    u_mesh: (pyshader.RES_UNIFORM, (1, 0), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 1), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 2), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord))
    color = (color - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = color  # noqa - shader output


@python2shader
def fragment_shader_phong(
    u_mesh: (pyshader.RES_UNIFORM, (1, 0), MeshBasicMaterial.uniform_type),
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_view: (pyshader.RES_INPUT, 2, vec3),
    v_light: (pyshader.RES_INPUT, 3, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    # todo: configure lights, and multiple lights
    # todo: allow configuring material specularity

    # Base colors
    albeido = u_mesh.color.rgb
    light_color = vec3(1, 1, 1)

    # Scale factors
    ambient_factor = 0.1
    diffuse_factor = 0.7
    specular_factor = 0.3

    # Base vectors
    normal = normalize(v_normal)
    view = normalize(v_view)
    light = normalize(v_light)

    # Maybe flip the normal - otherwise backfacing faces are not lit
    dotted = view.x * normal.x + view.y * normal.y + view.z * normal.z
    normal = mix(normal, -normal, f32(dotted < 0.0))

    # Ambient
    ambient_color = light_color * ambient_factor

    # Diffuse (blinn-phong light model)
    # lambert_term = clamp(dot(light, normal), 0.0, 1.0)
    dotted = light.x * normal.x + light.y * normal.y + light.z * normal.z
    lambert_term = clamp(dotted, 0.0, 1.0)
    diffuse_color = diffuse_factor * light_color * lambert_term

    # Specular
    shininess = 16.0
    halfway = normalize(light + view)  # halfway vector
    dotted = halfway.x * normal.x + halfway.y * normal.y + halfway.z * normal.z
    specular_term = clamp(dotted, 0.0, 1.0) ** shininess
    specular_color = specular_factor * specular_term * light_color

    # Put together
    final_color = albeido * (ambient_color + diffuse_color) + specular_color
    out_color = vec4(final_color, u_mesh.color.a)  # noqa - shader output
