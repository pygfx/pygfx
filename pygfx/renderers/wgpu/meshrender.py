import wgpu  # only for flags/enums
import pyshader
from pyshader import python2shader
from pyshader import f32, i32, vec2, vec3, vec4, mat4, Array


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

    # Use index buffer if present on the geometry
    index_buffer = getattr(geometry, "index", None)
    index_buffer = index_buffer if isinstance(index_buffer, Buffer) else None

    if index_buffer:
        n = index_buffer.data.size
    else:
        n = len(geometry.positions.data)

    # Collect vertex buffers
    vertex_buffers = {}
    vertex_buffers[0] = geometry.positions
    if getattr(geometry, "texcoords", None) is not None:
        vertex_buffers[1] = geometry.texcoords
    else:
        # meh, we need to specify a buffer if the shader uses it
        vertex_buffers[1] = geometry.positions

    # Normals. Usually it'd be given. If not, we'll calculate it from the vertices.
    if getattr(geometry, "normals", None) is not None:
        normal_buffer = geometry.normals
    else:
        normal_data = normals_from_vertices(geometry.positions.data, index_buffer.data)
        normal_buffer = Buffer(normal_data, usage="vertex|storage")
    vertex_buffers[2] = normal_buffer

    bindings0 = {
        0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo_uniform),
        1: (wgpu.BindingType.uniform_buffer, wobject.uniform_buffer),
        2: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
    }
    bindings1 = {}

    if material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("material.map is present, but geometry has no texcoords")
        bindings1[0] = wgpu.BindingType.sampler, material.map
        bindings1[1] = wgpu.BindingType.sampled_texture, material.map
        if material.map.view_dim == "2d":
            pass  # ok!
        elif material.map.view_dim == "3d":
            vertex_shader = vertex_shader_mesh_3dtex

    # Collect texture and sampler
    if isinstance(material, MeshNormalMaterial):
        fragment_shader = fragment_shader_normals
    elif isinstance(material, MeshNormalLinesMaterial):
        topology = wgpu.PrimitiveTopology.line_list
        vertex_shader = vertex_shader_normal_lines
        fragment_shader = fragment_shader_simple
        bindings1[2] = wgpu.BindingType.readonly_storage_buffer, geometry.positions
        bindings1[3] = wgpu.BindingType.readonly_storage_buffer, normal_buffer
        vertex_buffers = {}
        index_buffer = None
        n = geometry.positions.nitems * 2
    elif isinstance(material, MeshSliceMaterial):
        topology = wgpu.PrimitiveTopology.triangle_list
        vertex_shader = vertex_shader_mesh_slice
        fragment_shader = fragment_shader_mesh_slice
        bindings1[2] = wgpu.BindingType.readonly_storage_buffer, geometry.index
        bindings1[3] = wgpu.BindingType.readonly_storage_buffer, geometry.positions
        vertex_buffers = {}
        index_buffer = None
        n = (geometry.index.nitems // 3) * 6
    elif isinstance(material, MeshPhongMaterial):
        fragment_shader = fragment_shader_phong
        if material.map is not None:
            if "rgba" in material.map.format:
                fragment_shader = fragment_shader_textured_rgba_phong
            else:
                raise ValueError(
                    "Meshes with phong shading and grayscale textures is not yet supported"
                )
    else:
        fragment_shader = fragment_shader_simple
        if material.map is not None:
            if material.map.view_dim == "2d":
                if "rgba" in material.map.format:
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
        bindings1[4] = wgpu.BindingType.readonly_storage_buffer, wobject.matrices
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

    # Put it together!
    return [
        {
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
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
):
    world_pos = u_wobject.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    ndc_to_world = matrix_inverse(
        u_stdinfo.cam_transform * u_stdinfo.projection_transform
    )

    normal_vec = u_wobject.world_transform * vec4(in_normal.xyz, 1.0)

    view_vec4 = ndc_to_world * vec4(0, 0, 1, 1)
    view_vec = normalize(view_vec4.xyz / view_vec4.w)

    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = in_texcoord  # noqa - shader output

    # Vectors for lighting, all in world coordinates
    v_normal = normal_vec  # noqa
    v_view = view_vec  # noqa
    v_light = view_vec  # noqa


# todo: *sigh* it looks like we do need some form of templating


@python2shader
def vertex_shader_mesh_3dtex(  # also, no normals and lights
    in_pos: (pyshader.RES_INPUT, 0, vec4),
    in_texcoord: (pyshader.RES_INPUT, 1, vec3),
    u_stdinfo: ("uniform", (0, 0), stdinfo_uniform_type),
    u_wobject: ("uniform", (0, 1), Mesh.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec3),
):
    world_pos = u_wobject.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = in_texcoord  # noqa - shader output


@python2shader
def vertex_shader_normal_lines(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    buf_pos: (pyshader.RES_BUFFER, (1, 2), Array(vec4)),
    buf_normal: (pyshader.RES_BUFFER, (1, 3), Array(f32)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
):

    r = index % 2
    i = (index - r) // 2

    pos = buf_pos[i].xyz
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
    in_pos: (pyshader.RES_INPUT, 0, vec4),
    in_texcoord: (pyshader.RES_INPUT, 1, vec2),
    in_normal: (pyshader.RES_INPUT, 2, vec3),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec2),
    v_normal: (pyshader.RES_OUTPUT, 1, vec3),
    v_view: (pyshader.RES_OUTPUT, 2, vec3),
    v_light: (pyshader.RES_OUTPUT, 3, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), InstancedMesh.uniform_type),
    b_matrices: (pyshader.RES_BUFFER, (1, 4), Array(mat4)),
):
    # Note the extra matrix for the instance
    submatrix = b_matrices[instance_id]
    world_pos = u_wobject.world_transform * submatrix * vec4(in_pos.xyz, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    ndc_to_world = matrix_inverse(
        u_stdinfo.cam_transform * u_stdinfo.projection_transform
    )

    normal_vec = u_wobject.world_transform * vec4(in_normal.xyz, 1.0)

    view_vec = ndc_to_world * vec4(0, 0, 1, 1)
    view_vec = normalize(view_vec.xyz / view_vec.w)

    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = in_texcoord  # noqa - shader output

    # Vectors for lighting, all in world coordinates
    v_normal = normal_vec  # noqa
    v_view = view_vec  # noqa
    v_light = view_vec  # noqa


@python2shader
def fragment_shader_simple(
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    """Just draw the fragment in the mesh's color."""
    out_color = u_mesh.color  # noqa - shader output


@python2shader
def fragment_shader_normals(
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    """Draws the mesh in a color derived from the normal."""
    v = normalize(v_normal) * 0.5 + 0.5
    out_color = vec4(v, 1.0)  # noqa - shader output


@python2shader
def fragment_shader_textured_gray(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    val = f32(t_tex.sample(s_sam, v_texcoord).r)
    val = (val - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = vec4(val, val, val, 1.0)  # noqa - shader output


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
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord))
    color = (color - u_mesh.clim[0]) / (u_mesh.clim[1] - u_mesh.clim[0])
    out_color = color  # noqa - shader output


@python2shader
def fragment_shader_phong(
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_view: (pyshader.RES_INPUT, 2, vec3),
    v_light: (pyshader.RES_INPUT, 3, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
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


@python2shader
def fragment_shader_textured_rgba_phong(
    v_texcoord: (pyshader.RES_INPUT, 0, vec2),
    u_mesh: (pyshader.RES_UNIFORM, (0, 2), MeshBasicMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "2d i32"),
    v_normal: (pyshader.RES_INPUT, 1, vec3),
    v_view: (pyshader.RES_INPUT, 2, vec3),
    v_light: (pyshader.RES_INPUT, 3, vec3),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
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


# %% Mesh slice


@python2shader
def vertex_shader_mesh_slice(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), MeshSliceMaterial.uniform_type),
    buf_indices: (pyshader.RES_BUFFER, (1, 2), Array(i32)),
    buf_positions: (pyshader.RES_BUFFER, (1, 3), Array(vec4)),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_dist2center: (pyshader.RES_OUTPUT, 0, vec2),
    v_segment_length: (pyshader.RES_OUTPUT, 1, f32),
    v_segment_width: (pyshader.RES_OUTPUT, 2, f32),
):
    # This vertex shader uses VertexId and storage buffers instead of
    # vertex buffers. It creates 6 vertices for each face in the mesh,
    # drawn with triangle-list. For the faces that cross the plane, we
    # draw a (thick) line segment with round caps. Other faces become
    # degenerate triangles.

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
    pos1 = buf_positions[i1].xyz
    pos2 = buf_positions[i2].xyz
    pos3 = buf_positions[i3].xyz

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

    # Get the positions where the frame intersects the plane
    pos12, pos23, pos31 = mix(pos1, pos2, t1), mix(pos2, pos3, t2), mix(pos3, pos1, t3)
    pos00 = pos1

    # Selectors
    b1 = i32(t1 > 0.0) * i32(t1 < 1.0) * 4
    b2 = i32(t2 > 0.0) * i32(t2 < 1.0) * 2
    b3 = i32(t3 > 0.0) * i32(t3 < 1.0) * 1

    # b1+b2+b3     000    001    010    011    100    101    110    111
    positions_a = [pos00, pos00, pos00, pos23, pos00, pos12, pos12, pos12]
    positions_b = [pos00, pos00, pos00, pos31, pos00, pos31, pos23, pos23]

    # Select the two positions that define the line segment
    pos_index = b1 + b2 + b3
    pos_a = positions_a[pos_index]
    pos_b = positions_b[pos_index]

    if pos_index == 0:  # or n@u == 0
        # Just return the same vertex, resulting in degenerate triangles
        wpos1 = u_wobject.world_transform * vec4(pos1, 1.0)
        the_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos1
        the_coord = vec2(0, 0)
        segment_length = 0.0

    else:
        # Go from local coordinates to NDC
        wpos_a = u_wobject.world_transform * vec4(pos_a, 1.0)
        wpos_b = u_wobject.world_transform * vec4(pos_b, 1.0)
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
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), Mesh.uniform_type),
    u_material: (pyshader.RES_UNIFORM, (0, 2), MeshSliceMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
    out_depth: (pyshader.RES_OUTPUT, "FragDepth", f32),
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
