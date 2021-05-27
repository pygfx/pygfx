import wgpu  # only for flags/enums
import pyshader
from pyshader import python2shader
from pyshader import f32, i32, vec2, vec3, vec4, ivec2, ivec4, mat4, Array


from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Mesh, InstancedMesh
from ...materials import (
    MeshBasicMaterial,
    MeshNormalMaterial,
    MeshNormalLinesMaterial,
    MeshPhongMaterial,
    MeshSliceMaterial,
)
from ...resources import Buffer, Texture, TextureView
from ...utils import normals_from_vertices


@register_wgpu_render_function(Mesh, MeshBasicMaterial)
def mesh_renderer(wobject, render_info):
    """Render function capable of rendering meshes."""

    geometry = wobject.geometry
    material = wobject.material  # noqa

    # Initialize some pipeline things
    topology = wgpu.PrimitiveTopology.triangle_list
    vertex_shader = vertex_shader_mesh

    # We're assuming the presence of an index buffer for now
    assert getattr(geometry, "index")
    n = geometry.index.data.size

    # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
    if getattr(geometry, "normals", None) is not None:
        normal_buffer = geometry.normals
    else:
        normal_data = normals_from_vertices(
            geometry.positions.data, geometry.index.data
        )
        normal_buffer = Buffer(normal_data, usage="vertex|storage")

    # Init bindings 0: uniforms
    bindings0 = {
        0: ("buffer/uniform", render_info.stdinfo_uniform),
        1: ("buffer/uniform", wobject.uniform_buffer),
        2: ("buffer/uniform", material.uniform_buffer),
    }

    # We're using storage buffers for everything; no vertex nor index buffers.
    vertex_buffers = {}
    index_buffer = None

    # Init bindings 1: storage buffers, textures, and samplers
    bindings1 = {}
    bindings1[2] = "buffer/read_only_storage", geometry.index
    bindings1[3] = "buffer/read_only_storage", geometry.positions
    bindings1[4] = "buffer/read_only_storage", normal_buffer
    if getattr(geometry, "texcoords", None) is not None:
        bindings1[5] = "buffer/read_only_storage", geometry.texcoords

    if material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("material.map is present, but geometry has no texcoords")
        bindings1[0] = "sampler/filtering", material.map
        bindings1[1] = "texture/auto", material.map
        if material.map.view_dim == "2d":
            pass  # ok!
        elif material.map.view_dim == "3d":
            vertex_shader = vertex_shader_mesh_3dtex

    fs_entry_point = "fs_simple"

    # Collect texture and sampler
    if isinstance(material, MeshNormalMaterial):
        fragment_shader = fragment_shader_normals
    elif isinstance(material, MeshNormalLinesMaterial):
        topology = wgpu.PrimitiveTopology.line_list
        vertex_shader = vertex_shader_normal_lines
        fragment_shader = fragment_shader_simple
        bindings1[2] = "buffer/read_only_storage", geometry.positions
        bindings1[3] = "buffer/read_only_storage", normal_buffer
        vertex_buffers = {}
        index_buffer = None
        n = geometry.positions.nitems * 2
    elif isinstance(material, MeshSliceMaterial):
        topology = wgpu.PrimitiveTopology.triangle_list
        vertex_shader = vertex_shader_mesh_slice
        fragment_shader = fragment_shader_mesh_slice
        bindings1[2] = "buffer/read_only_storage", geometry.index
        bindings1[3] = "buffer/read_only_storage", geometry.positions
        vertex_buffers = {}
        index_buffer = None
        # n = (geometry.index.nitems // 3) * 6  # but what if data was nx3?
        n = (geometry.index.data.size // 3) * 6
    elif isinstance(material, MeshPhongMaterial):
        fragment_shader = fragment_shader_phong
        if material.map is not None:
            fs_entry_point = "fs_textured_rgba_phong"
            if "rgb" in material.map.format:  # rgb maps to rgba
                fragment_shader = fragment_shader_textured_rgba_phong
            else:
                raise ValueError(
                    "Meshes with phong shading and grayscale textures is not yet supported"
                )
    else:
        fragment_shader = fragment_shader_simple
        if material.map is not None:
            fs_entry_point = "fs_textured_rgba"
            if material.map.view_dim == "2d":
                if "rgb" in material.map.format:
                    fragment_shader = fragment_shader_textured_rgba
                else:
                    fragment_shader = fragment_shader_textured_gray
            elif material.map.view_dim == "3d":
                fragment_shader = fragment_shader_textured_gray_3dtex

    # Instanced meshes have their own vertex shader
    n_instances = 1
    if isinstance(wobject, InstancedMesh):
        if vertex_shader is not vertex_shader_mesh:
            raise TypeError(f"Instanced mesh does not work with {material}")
        vertex_shader = vertex_shader_mesh_instanced
        bindings1[6] = "buffer/read_only_storage", wobject.matrices
        n_instances = wobject.matrices.nitems

    # Use a version of the shader for float textures if necessary
    if material.map is not None:
        if "float" in material.map.format:
            if not hasattr(fragment_shader, "float_version"):
                func = fragment_shader.input
                tex_anno = func.__annotations__["t_tex"]
                func.__annotations__["t_tex"] = tex_anno[:2] + ("2d f32",)
                fragment_shader.float_version = python2shader(func)
                func.__annotations__["t_tex"] = tex_anno
            fragment_shader = fragment_shader.float_version

    wgsl = vertex_shader_mesh_wgsl

    # Put it together!
    return [
        {
            "shader": wgsl,
            "vs_entry_point": "vs_main",
            "fs_entry_point": fs_entry_point,
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": topology,
            "indices": (range(n), range(n_instances)),
            "index_buffer": index_buffer,
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


# %% Shaders


# Below is a vertex shader using vertex buffers. This is likely more
# efficient, but then we don't have the info we need to support picking :(
# We could consider adding special picking materials that are rendered to
# the pick texture in a separate render pass, or being able to turn off picking.
# Anyway, I don't expect the performance loss is significant in most cases ...
#
# @python2shader
# def vertex_shader_mesh(
#     index: (pyshader.RES_INPUT, "VertexId", "i32"),
#     in_pos: (pyshader.RES_INPUT, 0, vec3),
#     in_texcoord: (pyshader.RES_INPUT, 1, vec2),
#     in_normal: (pyshader.RES_INPUT, 2, vec3),
#     out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
#     v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
#     v_normal: (pyshader.RES_OUTPUT, 1, vec3),
#     v_view: (pyshader.RES_OUTPUT, 2, vec3),
#     v_light: (pyshader.RES_OUTPUT, 3, vec3),
#     u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
#     u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
# ):
#     world_pos = u_wobject.world_transform * vec4(in_pos.xyz, 1.0)
#     ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos
#
#     ndc_to_world = matrix_inverse(
#         u_stdinfo.cam_transform * u_stdinfo.projection_transform
#     )
#
#     normal_vec = u_wobject.world_transform * vec4(in_normal.xyz, 1.0)
#
#     view_vec4 = ndc_to_world * vec4(0, 0, 1, 1)
#     view_vec = normalize(view_vec4.xyz / view_vec4.w)
#
#     out_pos = ndc_pos  # noqa - shader output
#     v_texcoord = in_texcoord  # noqa - shader output
#
#     # Vectors for lighting, all in world coordinates
#     v_normal = normal_vec  # noqa
#     v_view = view_vec  # noqa
#     v_light = view_vec  # noqa


@python2shader
def vertex_shader_mesh(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_indices: (pyshader.RES_BUFFER, (1, 2), Array(i32)),
    buf_pos: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    buf_normal: (pyshader.RES_BUFFER, (1, 4), Array(f32)),
    buf_texcoord: (pyshader.RES_BUFFER, (1, 5), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
    v_normal: (pyshader.RES_OUTPUT, 1, vec3),
    v_view: (pyshader.RES_OUTPUT, 2, vec3),
    v_light: (pyshader.RES_OUTPUT, 3, vec3),
    v_face_idx: (pyshader.RES_OUTPUT, 4, vec4),
    v_face_coords: (pyshader.RES_OUTPUT, 5, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
):

    # Select what face we're at
    face_index = index // 3
    sub_index = index % 3
    i1 = buf_indices[face_index * 3 + 0]
    i2 = buf_indices[face_index * 3 + 1]
    i3 = buf_indices[face_index * 3 + 2]
    i0 = [i1, i2, i3][sub_index]

    # Vertex positions of this face, in local object coordinates
    raw_pos = vec3(buf_pos[i0 * 3 + 0], buf_pos[i0 * 3 + 1], buf_pos[i0 * 3 + 2])
    world_pos = u_wobject.world_transform * vec4(raw_pos, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    ndc_to_world = matrix_inverse(
        u_stdinfo.cam_transform * u_stdinfo.projection_transform
    )

    normal = vec3(
        buf_normal[i0 * 3 + 0], buf_normal[i0 * 3 + 1], buf_normal[i0 * 3 + 2]
    )
    normal = (u_wobject.world_transform * vec4(normal.xyz, 1.0)).xyz

    view_vec4 = ndc_to_world * vec4(0, 0, 1, 1)
    view_vec = normalize(view_vec4.xyz / view_vec4.w)

    # Vectors for lighting, all in world coordinates
    v_normal = normal  # noqa
    v_view = view_vec  # noqa
    v_light = view_vec  # noqa

    texcoord = vec2(buf_texcoord[i0 * 2 + 0], buf_texcoord[i0 * 2 + 1])
    v_texcoord = texcoord  # noqa - shader output

    out_pos = ndc_pos  # noqa - shader output

    # Set varying for picking. We store the face_index, and 3 weights
    # that indicate how close the fragment is to each vertex (barycentric
    # coordinates). This allows the selection of the nearest vertex or
    # edge. Note that integers larger than about 4M loose too much
    # precision when passed as a varyings (on my machine). We therefore
    # encode them in two values.
    v_face_idx = vec4(0.0, 0.0, face_index // 10000, face_index % 10000)  # noqa
    v_face_coords = [vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)][sub_index]  # noqa


vertex_shader_mesh_wgsl = """

struct VertexInput {
    [[builtin(vertex_index)]] index : u32;
};

struct VertexOutput {
    [[location(0)]] texcoord: vec2<f32>;
    [[location(1)]] normal: vec3<f32>;
    [[location(2)]] view: vec3<f32>;
    [[location(3)]] light: vec3<f32>;
    [[location(4)]] face_idx: vec4<f32>;
    [[location(5)]] face_coords: vec3<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] pick: vec4<i32>;
};

[[block]]
struct Stdinfo {
    cam_transform: mat4x4<f32>;
    cam_transform_inv: mat4x4<f32>;
    projection_transform: mat4x4<f32>;
    projection_transform_inv: mat4x4<f32>;
    physical_size: vec2<f32>;
    logical_size: vec2<f32>;
};

[[block]]
struct Wobject {
    world_transform: mat4x4<f32>;
    id: i32;
};

[[block]]
struct Material {
    color: vec4<f32>;
    clim: vec2<f32>;
};

[[block]]
struct BufferI32 {
    data: [[stride(4)]] array<i32>;
};

[[block]]
struct BufferF32 {
    data: [[stride(4)]] array<f32>;
};

// ----- Uniforms

[[group(0), binding(0)]]
var u_stdinfo: Stdinfo;

[[group(0), binding(1)]]
var u_wobject: Wobject;

[[group(0), binding(2)]]
var u_mat: Material;

// ----- Storage buffers/textures

[[group(1), binding(0)]]
var r_sampler: sampler;

[[group(1), binding(1)]]
var r_tex: texture_2d<f32>;

[[group(1), binding(2)]]
var<storage> s_indices: [[access(read)]] BufferI32;

[[group(1), binding(3)]]
var<storage> s_pos: [[access(read)]] BufferF32;

[[group(1), binding(4)]]
var<storage> s_normal: [[access(read)]] BufferF32;

[[group(1), binding(5)]]
var<storage> s_texcoord: [[access(read)]] BufferF32;

// ----- Shaders

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {

    // Select what face we're at
    let index = i32(in.index);
    let face_index = index / 3;
    let sub_index = index % 3;
    let i1 = s_indices.data[face_index * 3 + 0];
    let i2 = s_indices.data[face_index * 3 + 1];
    let i3 = s_indices.data[face_index * 3 + 2];
    let i0 = array<i32, 3>(i1, i2, i3)[sub_index];

    // Vertex positions of this face, in local object coordinates
    let raw_pos = vec3<f32>(s_pos.data[i0 * 3 + 0], s_pos.data[i0 * 3 + 1], s_pos.data[i0 * 3 + 2]);
    let world_pos = u_wobject.world_transform * vec4<f32>(raw_pos, 1.0);
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    //let ndc_to_world = matrix_inverse(u_stdinfo.cam_transform * u_stdinfo.projection_transform);
    let ndc_to_world = u_stdinfo.projection_transform_inv * u_stdinfo.cam_transform_inv;

    // Prepare output
    var out: VertexOutput;

    // Set position and texcoords
    out.pos = vec4<f32>(ndc_pos.xyz, ndc_pos.w);
    out.texcoord = vec2<f32>(s_texcoord.data[i0 * 2 + 0], s_texcoord.data[i0 * 2 + 1]);

    // Vectors for lighting, all in world coordinates

    let normal_ = vec3<f32>(
        s_normal.data[i0 * 3 + 0], s_normal.data[i0 * 3 + 1], s_normal.data[i0 * 3 + 2]
    );
    out.normal = (u_wobject.world_transform * vec4<f32>(normal_.xyz, 1.0)).xyz;

    let view_vec4 = ndc_to_world * vec4<f32>(0.0, 0.0, 1.0, 1.0);
    let view_vec = normalize(view_vec4.xyz / view_vec4.w);
    out.view = view_vec;
    out.light = view_vec;

    // Set varyings for picking. We store the face_index, and 3 weights
    // that indicate how close the fragment is to each vertex (barycentric
    // coordinates). This allows the selection of the nearest vertex or
    // edge. Note that integers larger than about 4M loose too much
    // precision when passed as a varyings (on my machine). We therefore
    // encode them in two values.
    out.face_idx = vec4<f32>(0.0, 0.0, f32(face_index / 10000), f32(face_index % 10000));
    out.face_coords = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0))[sub_index];

    return out;
}


[[stage(fragment)]]
fn fs_simple(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = u_mat.color;
    let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
    let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
    out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);
    return out;
}


[[stage(fragment)]]
fn fs_textured_rgba(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let color = textureSample(r_tex, r_sampler, in.texcoord.xy);
    out.color = (color - u_mat.clim[0]) / (u_mat.clim[1] - u_mat.clim[0]);

    let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
    let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
    out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);
    return out;
}


[[stage(fragment)]]
fn fs_textured_rgba_phong(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let color_sampled = textureSample(r_tex, r_sampler, in.texcoord.xy);
    //let texcoords_u = vec2<i32>(in.texcoord * vec2<f32>(textureDimensions(r_tex)) % vec2<f32>(textureDimensions(r_tex)));
    //let color_sampled = vec4<f32>(textureLoad(r_tex, texcoords_u, 0));
    let color = (color_sampled - u_mat.clim[0]) / (u_mat.clim[1] - u_mat.clim[0]);

    // Base colors
    let albeido = color.rgb;
    let light_color = vec3<f32>(1.0, 1.0, 1.0);

    // Parameters
    // todo: allow configuring material specularity
    let ambient_factor = 0.1;
    let diffuse_factor = 0.7;
    let specular_factor = 0.3;
    let shininess = 16.0;

    // Base vectors
    var normal: vec3<f32> = normalize(in.normal);
    let view = normalize(in.view);
    let light = normalize(in.light);

    // Maybe flip the normal - otherwise backfacing faces are not lit
    normal = select(normal, -normal, dot(view, normal) >= 0.0);

    // Ambient
    let ambient_color = light_color * ambient_factor;

    // Diffuse (blinn-phong light model)
    let lambert_term = clamp(dot(light, normal), 0.0, 1.0);
    let diffuse_color = diffuse_factor * light_color * lambert_term;

    // Specular
    let halfway = normalize(light + view);  // halfway vector
    let specular_term = pow(clamp(dot(halfway,  normal), 0.0, 1.0), shininess);
    let specular_color = specular_factor * specular_term * light_color;

    // Put together
    let final_color = albeido * (ambient_color + diffuse_color) + specular_color;
    out.color = vec4<f32>(final_color, color.a);

    // Picking
    let face_id = vec2<i32>(in.face_idx.xz * 10000.0 + in.face_idx.yw + 0.5);  // inst+face
    let w8 = vec3<i32>(in.face_coords.xyz * 255.0 + 0.5);
    out.pick = vec4<i32>(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z);

    return out;
}

"""

# todo: *sigh* it looks like we do need some form of templating


@python2shader
def vertex_shader_mesh_3dtex(  # also, no normals and lights
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_indices: (pyshader.RES_BUFFER, (1, 2), Array(i32)),
    buf_pos: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    buf_texcoord: (pyshader.RES_BUFFER, (1, 5), Array(f32)),
    u_stdinfo: ("uniform", (0, 0), stdinfo_uniform_type),
    u_wobject: ("uniform", (0, 1), Mesh.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec3),
):
    # Select what face we're at
    face_index = index // 3
    sub_index = index % 3
    i1 = buf_indices[face_index * 3 + 0]
    i2 = buf_indices[face_index * 3 + 1]
    i3 = buf_indices[face_index * 3 + 2]
    i0 = [i1, i2, i3][sub_index]

    raw_pos = vec3(buf_pos[i0 * 3 + 0], buf_pos[i0 * 3 + 1], buf_pos[i0 * 3 + 2])
    world_pos = u_wobject.world_transform * vec4(raw_pos, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos
    out_pos = ndc_pos  # noqa - shader output

    texcoord = vec3(
        buf_texcoord[i0 * 3 + 0], buf_texcoord[i0 * 3 + 1], buf_texcoord[i0 * 3 + 2]
    )
    v_texcoord = texcoord  # noqa - shader output


@python2shader
def vertex_shader_normal_lines(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 2), Array(f32)),
    buf_normal: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
):

    r = index % 2
    i = (index - r) // 2

    pos = vec3(buf_pos[i * 3 + 0], buf_pos[i * 3 + 1], buf_pos[i * 3 + 2])
    normal = vec3(buf_normal[i * 3 + 0], buf_normal[i * 3 + 1], buf_normal[i * 3 + 2])

    world_pos1 = u_wobject.world_transform * vec4(pos, 1.0)
    world_pos2 = u_wobject.world_transform * vec4(pos + normal, 1.0)

    # The normal is sized in world coordinates
    world_normal = normalize(world_pos2 - world_pos1)

    world_pos = world_pos1 + f32(r) * world_normal * 1.0
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos
    out_pos = ndc_pos  # noqa - shader output


@python2shader
def vertex_shader_mesh_instanced(
    instance_id: (pyshader.RES_INPUT, "InstanceId", i32),
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    buf_indices: (pyshader.RES_BUFFER, (1, 2), Array(i32)),
    buf_pos: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    buf_normal: (pyshader.RES_BUFFER, (1, 4), Array(f32)),
    buf_texcoord: (pyshader.RES_BUFFER, (1, 5), Array(f32)),
    buf_matrices: (pyshader.RES_BUFFER, (1, 6), Array(mat4)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
    v_normal: (pyshader.RES_OUTPUT, 1, vec3),
    v_view: (pyshader.RES_OUTPUT, 2, vec3),
    v_light: (pyshader.RES_OUTPUT, 3, vec3),
    v_face_idx: (pyshader.RES_OUTPUT, 4, vec4),
    v_face_coords: (pyshader.RES_OUTPUT, 5, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), InstancedMesh.uniform_type),
):
    # Select matrix for this instance
    submatrix = buf_matrices[instance_id]

    # Select what face we're at
    face_index = index // 3
    sub_index = index % 3
    i1 = buf_indices[face_index * 3 + 0]
    i2 = buf_indices[face_index * 3 + 1]
    i3 = buf_indices[face_index * 3 + 2]
    i0 = [i1, i2, i3][sub_index]

    # Vertex positions of this face, in local object coordinates
    raw_pos = vec3(buf_pos[i0 * 3 + 0], buf_pos[i0 * 3 + 1], buf_pos[i0 * 3 + 2])
    world_pos = u_wobject.world_transform * submatrix * vec4(raw_pos, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    ndc_to_world = matrix_inverse(
        u_stdinfo.cam_transform * u_stdinfo.projection_transform
    )

    normal = vec3(
        buf_normal[i0 * 3 + 0], buf_normal[i0 * 3 + 1], buf_normal[i0 * 3 + 2]
    )
    normal = (u_wobject.world_transform * vec4(normal.xyz, 1.0)).xyz

    view_vec4 = ndc_to_world * vec4(0, 0, 1, 1)
    view_vec = normalize(view_vec4.xyz / view_vec4.w)

    # Vectors for lighting, all in world coordinates
    v_normal = normal  # noqa
    v_view = view_vec  # noqa
    v_light = view_vec  # noqa

    texcoord = vec2(buf_texcoord[i0 * 2 + 0], buf_texcoord[i0 * 2 + 1])
    v_texcoord = texcoord  # noqa - shader output

    out_pos = ndc_pos  # noqa - shader output

    # Set varying for picking. We store the face_index, and 3 weights
    face_idx = vec4(
        instance_id // 10000,
        instance_id % 10000,
        face_index // 10000,
        face_index % 10000,
    )
    v_face_idx = face_idx  # noqa
    v_face_coords = [vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)][sub_index]  # noqa


@python2shader
def fragment_shader_simple(
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    """Just draw the fragment in the mesh's color."""
    out_color = u_mesh.color  # noqa - shader output

    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


@python2shader
def fragment_shader_normals(
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    """Draws the mesh in a color derived from the normal."""
    v = normalize(v_normal) * 0.5 + 0.5
    out_color = vec4(v, 1.0)  # noqa - shader output

    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


@python2shader
def fragment_shader_textured_gray(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    val = f32(t_tex.sample(s_sam, v_texcoord).r)
    val = (val - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = vec4(val, val, val, 1.0)  # noqa - shader output

    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


@python2shader
def fragment_shader_textured_gray_3dtex(
    v_texcoord: (pyshader.RES_INPUT, 0, vec3),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "3d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    val = f32(t_tex.sample(s_sam, v_texcoord).r)
    val = (val - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = vec4(val, val, val, 1.0)  # noqa - shader output


@python2shader
def fragment_shader_textured_rgba(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord))
    color = (color - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = color  # noqa - shader output

    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


@python2shader
def fragment_shader_phong(
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_view: (pyshader.RES_INPUT, 2, vec3),
    v_light: (pyshader.RES_INPUT, 3, vec3),
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    # todo: configure lights, and multiple lights

    # Base colors
    albeido = u_mesh.color.rgb
    light_color = vec3(1, 1, 1)

    # Parameters
    # todo: allow configuring material specularity
    ambient_factor = 0.1
    diffuse_factor = 0.7
    specular_factor = 0.3
    shininess = 16.0

    # Base vectors
    normal = normalize(v_normal)
    view = normalize(v_view)
    light = normalize(v_light)

    # Maybe flip the normal - otherwise backfacing faces are not lit
    normal = mix(normal, -normal, f32(view @ normal < 0.0))

    # Ambient
    ambient_color = light_color * ambient_factor

    # Diffuse (blinn-phong light model)
    lambert_term = clamp(light @ normal, 0.0, 1.0)
    diffuse_color = diffuse_factor * light_color * lambert_term

    # Specular
    halfway = normalize(light + view)  # halfway vector
    specular_term = clamp(halfway @ normal, 0.0, 1.0) ** shininess
    specular_color = specular_factor * specular_term * light_color

    # Put together
    final_color = albeido * (ambient_color + diffuse_color) + specular_color
    out_color = vec4(final_color, u_mesh.color.a)  # noqa - shader output

    # The picking output consists of the wobject id, the face_index, and the
    # face_weights (the weights are encoded into a single int32).
    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


@python2shader
def fragment_shader_textured_rgba_phong(
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_view: (pyshader.RES_INPUT, 2, vec3),
    v_light: (pyshader.RES_INPUT, 3, vec3),
    v_face_idx: (pyshader.RES_INPUT, 4, vec4),
    v_face_coords: (pyshader.RES_INPUT, 5, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord))
    color = (color - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])

    # Base colors
    albeido = color.rgb
    light_color = vec3(1, 1, 1)

    # Parameters
    # todo: allow configuring material specularity
    ambient_factor = 0.1
    diffuse_factor = 0.7
    specular_factor = 0.3
    shininess = 16.0

    # Base vectors
    normal = normalize(v_normal)
    view = normalize(v_view)
    light = normalize(v_light)

    # Maybe flip the normal - otherwise backfacing faces are not lit
    normal = mix(normal, -normal, f32(view @ normal < 0.0))

    # Ambient
    ambient_color = light_color * ambient_factor

    # Diffuse (blinn-phong light model)
    lambert_term = clamp(light @ normal, 0.0, 1.0)
    diffuse_color = diffuse_factor * light_color * lambert_term

    # Specular
    halfway = normalize(light + view)  # halfway vector
    specular_term = clamp(halfway @ normal, 0.0, 1.0) ** shininess
    specular_color = specular_factor * specular_term * light_color

    # Put together
    final_color = albeido * (ambient_color + diffuse_color) + specular_color
    out_color = vec4(final_color, color.a)  # noqa - shader output

    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa


# %% Mesh slice


@python2shader
def vertex_shader_mesh_slice(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), MeshSliceMaterial.uniform_type),
    buf_indices: (pyshader.RES_BUFFER, (1, 2), Array(i32)),
    buf_pos: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_dist2center: (pyshader.RES_OUTPUT, 0, vec2),
    v_segment_length: (pyshader.RES_OUTPUT, 1, f32),
    v_segment_width: (pyshader.RES_OUTPUT, 2, f32),
    v_face_idx: (pyshader.RES_OUTPUT, 3, vec4),
    v_face_coords: (pyshader.RES_OUTPUT, 4, vec3),
):
    # This vertex shader uses VertexId and storage buffers instead of
    # vertex buffers. It creates 6 vertices for each face in the mesh,
    # drawn with triangle-list. For the faces that cross the plane, we
    # draw a (thick) line segment with round caps (we need 6 verts for that).
    # Other faces become degenerate triangles.

    # Prepare some numbers
    screen_factor = u_stdinfo.logical_size.xy / 2.0
    l2p = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    line_width = u_material.thickness  # in logical pixels

    # Get the face index, and sample the vertex indices
    segment_index = index % 6
    face_index = (index - segment_index) // 6
    i1 = buf_indices[face_index * 3 + 0]
    i2 = buf_indices[face_index * 3 + 1]
    i3 = buf_indices[face_index * 3 + 2]

    # Vertex positions of this face, in local object coordinates
    pos1 = vec3(buf_pos[i1 * 3 + 0], buf_pos[i1 * 3 + 1], buf_pos[i1 * 3 + 2])
    pos2 = vec3(buf_pos[i2 * 3 + 0], buf_pos[i2 * 3 + 1], buf_pos[i2 * 3 + 2])
    pos3 = vec3(buf_pos[i3 * 3 + 0], buf_pos[i3 * 3 + 1], buf_pos[i3 * 3 + 2])
    pos1_ = u_wobject.world_transform * vec4(pos1, 1.0)
    pos2_ = u_wobject.world_transform * vec4(pos2, 1.0)
    pos3_ = u_wobject.world_transform * vec4(pos3, 1.0)
    pos1 = pos1_.xyz / pos1_.w
    pos2 = pos2_.xyz / pos2_.w
    pos3 = pos3_.xyz / pos3_.w

    # Get the plane definition
    plane = u_material.plane.xyzw  # ax + by + cz + d
    n = plane.xyz  # not necessarily a unit vector

    # Intersect the plane with pos 1 and 2
    p, u = pos1.xyz, pos2.xyz - pos1.xyz
    t1 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / (n @ u)
    # Intersect the plane with pos 2 and 3
    p, u = pos2.xyz, pos3.xyz - pos2.xyz
    t2 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / (n @ u)
    # Intersect the plane with pos 3 and 1
    p, u = pos3.xyz, pos1.xyz - pos3.xyz
    t3 = -(plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w) / (n @ u)

    # Selectors
    b1 = i32(t1 > 0.0) * i32(t1 < 1.0) * 4
    b2 = i32(t2 > 0.0) * i32(t2 < 1.0) * 2
    b3 = i32(t3 > 0.0) * i32(t3 < 1.0) * 1
    pos_index = b1 + b2 + b3

    if pos_index < 3:  # or n@u == 0
        # Just return the same vertex, resulting in degenerate triangles
        wpos1 = vec4(pos1, 1.0)
        the_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos1
        the_coord = vec2(0, 0)
        segment_length = 0.0

    else:
        # Get the positions where the frame intersects the plane
        pos00 = pos1
        pos12 = mix(pos1, pos2, t1)
        pos23 = mix(pos2, pos3, t2)
        pos31 = mix(pos3, pos1, t3)
        # b1+b2+b3     000    001    010    011    100    101    110    111
        positions_a = [pos00, pos00, pos00, pos23, pos00, pos12, pos12, pos12]
        positions_b = [pos00, pos00, pos00, pos31, pos00, pos31, pos23, pos23]
        # Select the two positions that define the line segment
        pos_a = positions_a[pos_index]
        pos_b = positions_b[pos_index]

        # Same for face weights
        fw00 = vec3(0.5, 0.5, 0.5)
        fw12 = mix(vec3(1, 0, 0), vec3(0, 1, 0), t1)
        fw23 = mix(vec3(0, 1, 0), vec3(0, 0, 1), t2)
        fw31 = mix(vec3(0, 0, 1), vec3(1, 0, 0), t3)
        fws_a = [fw00, fw00, fw00, fw23, fw00, fw12, fw12, fw12]
        fws_b = [fw00, fw00, fw00, fw31, fw00, fw31, fw23, fw23]
        fw_a = fws_a[pos_index]
        fw_b = fws_b[pos_index]

        # Go from local coordinates to NDC
        wpos_a = vec4(pos_a, 1.0)
        wpos_b = vec4(pos_b, 1.0)
        npos_a = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos_a
        npos_b = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos_b
        # Don't forget to "normalize"!
        # todo: omitting this step diminish the line width with distance, but it that the way?
        npos_a = npos_a / npos_a.w
        npos_b = npos_b / npos_b.w

        # And to logical pixel coordinates (don't worry about offset)
        ppos_a = npos_a.xy * screen_factor
        ppos_b = npos_b.xy * screen_factor

        # Get the segment vector, its length, and how much it scales because of line width
        v0 = ppos_b - ppos_a
        segment_length = length(v0)  # in logical pixels
        segment_factor = (segment_length + line_width) / segment_length

        # Get the (orthogonal) unit vectors that span the segment
        v1 = normalize(v0)
        v2 = vec2(+v1.y, -v1.x)

        # Get the vector, in local logical pixels for the segment's square
        pvec_local = 0.5 * vec2(segment_length + line_width, line_width)

        # Select one of the four corners of the segment rectangle
        vecs = [
            vec2(-1, -1),
            vec2(+1, +1),
            vec2(-1, +1),
            vec2(+1, +1),
            vec2(-1, -1),
            vec2(+1, -1),
        ]
        the_vec = vecs[segment_index]

        # Construct the position, also make sure zw scales correctly
        pvec = the_vec.x * pvec_local.x * v1 + the_vec.y * pvec_local.y * v2
        zw_range = (npos_b.zw - npos_a.zw) * segment_factor
        the_pos_p = 0.5 * (ppos_a + ppos_b) + pvec
        the_pos_zw = 0.5 * (npos_a.zw + npos_b.zw) + the_vec.x * zw_range * 0.5
        the_pos = vec4(the_pos_p / screen_factor, the_pos_zw)

        # Define the local coordinate in physical pixels
        the_coord = the_vec * pvec_local

        # Picking info
        v_face_idx = vec4(0.0, 0.0, face_index // 10000, face_index % 10000)  # noqa
        v_face_coords = mix(fw_a, fw_b, the_vec.x * 0.5 + 0.5)  # noqa

    # Shader output
    out_pos = the_pos  # noqa
    v_dist2center = the_coord * l2p  # noqa
    v_segment_length = segment_length * l2p  # noqa
    v_segment_width = line_width * l2p  # noqa


@pyshader.python2shader
def fragment_shader_mesh_slice(
    in_coord: (pyshader.RES_INPUT, "FragCoord", vec4),
    v_dist2center: (pyshader.RES_INPUT, 0, vec2),
    v_segment_length: (pyshader.RES_INPUT, 1, f32),
    v_segment_width: (pyshader.RES_INPUT, 2, f32),
    v_face_idx: (pyshader.RES_INPUT, 3, vec4),
    v_face_coords: (pyshader.RES_INPUT, 4, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), MeshSliceMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_depth: (pyshader.RES_OUTPUT, "FragDepth", f32),
    out_pick: (pyshader.RES_OUTPUT, 1, ivec4),
):
    # Discart fragments that are too far from the centerline. This makes round caps.
    # Note that we operate in physical pixels here.
    distx = max(0.0, abs(v_dist2center.x) - 0.5 * v_segment_length)
    dist = length(vec2(distx, v_dist2center.y))
    if dist > v_segment_width * 0.5:
        return  # discard

    # No aa. This is something we need to decide on. See line renderer.
    alpha = 1.0

    # Set color
    color = u_material.color
    out_color = vec4(color.rgb, min(1.0, color.a) * alpha)  # noqa - shader assign

    # Set pick info
    face_id = ivec2(v_face_idx.xz * 10000.0 + v_face_idx.yw + 0.5)  # inst+face
    w8 = ivec3(v_face_coords.xyz * 255.0 + 0.5)
    out_pick = ivec4(u_wobject.id, face_id, w8.x * 65536 + w8.y * 256 + w8.z)  # noqa
