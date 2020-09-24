import wgpu  # only for flags/enums
import pyshader
from pyshader import python2shader
from pyshader import Array, f32, i32, vec3, vec4

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Volume
from ...materials import VolumeSliceMaterial
from ...resources import Texture, TextureView


@register_wgpu_render_function(Volume, VolumeSliceMaterial)
def volume_slice_renderer(wobject, render_info):
    """Render function capable of rendering volumes."""

    geometry = wobject.geometry
    material = wobject.material  # noqa

    fragment_shader = fragment_shader_textured_gray

    vertex_buffers = {}
    bindings0 = {
        0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo_uniform),
        1: (wgpu.BindingType.uniform_buffer, wobject.uniform_buffer),
        2: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
    }
    bindings1 = {
        0: (wgpu.BindingType.storage_buffer, geometry.positions),
        1: (wgpu.BindingType.storage_buffer, geometry.texcoords),
    }

    topology = wgpu.PrimitiveTopology.triangle_list
    n = 12

    # Collect texture and sampler
    if material.map is not None:
        if isinstance(material.map, TextureView):
            view = material.map
        elif isinstance(material.map, Texture):
            view = material.map.get_view(filter="linear")
        else:
            raise TypeError("material.map must be a TextureView")
        if view.view_dim.lower() != "3d":
            raise TypeError("material.map must a 3D texture (view)")
        elif getattr(geometry, "texcoords", None) is None:
            raise ValueError("material.map is present, but geometry has no texcoords")
        bindings1[2] = wgpu.BindingType.sampler, view
        bindings1[3] = wgpu.BindingType.sampled_texture, view
        # Use a version of the shader for float textures if necessary
        if "float" in view.format:
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
            "vertex_shader": vertex_shader_volume_slice,
            "fragment_shader": fragment_shader,
            "primitive_topology": topology,
            "indices": (range(n), range(1)),
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]


@python2shader
def vertex_shader_volume_slice(
    index: ("input", "VertexId", "i32"),
    # in_pos: (pyshader.RES_INPUT, 0, vec4),
    buf_positions: ("buffer", (1, 0), Array(vec4)),
    buf_texcoords: ("buffer", (1, 1), Array(f32)),
    u_stdinfo: ("uniform", (0, 0), stdinfo_uniform_type),
    u_wobject: ("uniform", (0, 1), Volume.uniform_type),
    u_material: ("uniform", (0, 2), VolumeSliceMaterial.uniform_type),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec3),
):
    # We're assuming a box geometry, using the same layout as a simple
    # ThreeJS BoxBufferGeometry. And we're only using the first eight
    # vertices. These are laid out like this:
    #
    #   Vertices       Planes (right, left, back, front, top, bottom)
    #                            0      1    2      3     4     5
    #
    #    5----0        0: 0231        +----+
    #   /|   /|        1: 7546       /|24 /|
    #  7----2 |        2: 5014      +----+ |0
    #  | 4--|-1        3: 2763     1| +--|-+
    #  |/   |/         4: 0572      |/35 |/
    #  6----3          5: 3641      +----+

    ##

    plane = u_material.plane.xyzw  # ax + by + cz + d
    n = plane.xyz

    # Define edges (using vertex indices), and their matching plane
    # indices (each edge touches two planes). Note that these need to
    # match the above figure, and that needs to match with the actual
    # BoxGeometry implementation!
    edges = [
        [0, 2],
        [2, 3],
        [3, 1],
        [1, 0],
        [4, 6],
        [6, 7],
        [7, 5],
        [5, 4],
        [5, 0],
        [1, 4],
        [2, 7],
        [6, 3],
    ]
    ed2pl = [
        [0, 4],
        [0, 3],
        [0, 5],
        [0, 2],
        [1, 5],
        [1, 3],
        [1, 4],
        [1, 2],
        [2, 4],
        [2, 5],
        [3, 4],
        [3, 5],
    ]

    # Init intersection info
    intersect_flags = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    intersect_positions = [
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
    ]
    intersect_texcoords = [
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
    ]

    # Intersect the 12 edges
    for i in range(12):
        edge = edges[i]
        p1 = buf_positions[edge[0]].xyz
        p2 = buf_positions[edge[1]].xyz
        p1_ = u_wobject.world_transform * vec4(p1, 1.0)
        p2_ = u_wobject.world_transform * vec4(p2, 1.0)
        p1 = p1_.xyz / p1_.w
        p2 = p2_.xyz / p2_.w
        tc1 = vec3(
            buf_texcoords[edge[0] * 3],
            buf_texcoords[edge[0] * 3 + 1],
            buf_texcoords[edge[0] * 3 + 2],
        )
        tc2 = vec3(
            buf_texcoords[edge[1] * 3],
            buf_texcoords[edge[1] * 3 + 1],
            buf_texcoords[edge[1] * 3 + 2],
        )
        u = p2 - p1
        t = -(plane.x * p1.x + plane.y * p1.y + plane.z * p1.z + plane.w) / (n @ u)
        intersects = i32(t > 0.0) * i32(t < 1.0)
        intersect_flags[i] = intersects
        intersect_positions[i] = mix(p1, p2, t)
        intersect_texcoords[i] = mix(tc1, tc2, t)

    # Init six vertices
    vertices = [
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
    ]
    texcoords = [
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
        vec3(0, 0, 0),
    ]

    # Find first intersection point. This can be any valid intersection.
    # In ed2pl[i][0], the 0 could also be a one. It would mean that we'd
    # move around the box in the other direction.
    plane_index = 0
    for i in range(12):
        if intersect_flags[i] == 1:
            plane_index = ed2pl[i][0]
            vertices[0] = intersect_positions[i]
            texcoords[0] = intersect_texcoords[i]
            break

    # From there take (at most) 5 steps
    i_start = i
    i_last = i
    max_iter = 6
    for iter in range(1, max_iter):
        for i in range(12):
            if i != i_last and intersect_flags[i] == 1:
                if ed2pl[i][0] == plane_index:
                    vertices[iter] = intersect_positions[i]
                    texcoords[iter] = intersect_texcoords[i]
                    plane_index = ed2pl[i][1]
                    i_last = i
                    break
                elif ed2pl[i][1] == plane_index:
                    vertices[iter] = intersect_positions[i]
                    texcoords[iter] = intersect_texcoords[i]
                    plane_index = ed2pl[i][0]
                    i_last = i
                    break
        if i_last == i_start:
            max_iter = iter
            break

    # Make the rest degenerate triangles
    for i in range(max_iter, 6):
        vertices[i] = vertices[0]

    # Now select the current vertex. We mimic a triangle fan with a triangle list.
    # This works whether the number of vertices/intersections is 3, 4, 5, and 6.
    indexmap = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]
    the_pos = vertices[indexmap[index]]
    the_tc = texcoords[indexmap[index]]

    world_pos = vec4(the_pos, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos
    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = the_tc  # noqa - shader output


@python2shader
def fragment_shader_textured_gray(
    v_texcoord: (pyshader.RES_INPUT, 0, vec3),
    u_material: (pyshader.RES_UNIFORM, (0, 2), VolumeSliceMaterial.uniform_type),
    s_sam: (pyshader.RES_SAMPLER, (1, 2), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 3), "3d i32"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    val = f32(t_tex.sample(s_sam, v_texcoord).r)
    val = (val - u_material.clim[0]) / (u_material.clim[1] - u_material.clim[0])
    out_color = vec4(val, val, val, 1.0)  # noqa - shader output
