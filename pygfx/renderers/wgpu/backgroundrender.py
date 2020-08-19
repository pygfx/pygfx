import wgpu  # only for flags/enums
import pyshader
from pyshader import python2shader
from pyshader import vec3, vec4

from . import register_wgpu_render_function, stdinfo_uniform_type, wobject_uniform_type
from ...objects import Background
from ...materials import BackgroundMaterial, BackgroundImageMaterial
from ...datawrappers import Texture, TextureView


@python2shader
def vertex_shader_simple(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec3),
):
    # Define positions at the four corners of the viewport, at the largest depth
    positions = [
        vec4(-1.0, -1.0, 1.0, 1.0),
        vec4(+1.0, -1.0, 1.0, 1.0),
        vec4(-1.0, +1.0, 1.0, 1.0),
        vec4(+1.0, +1.0, 1.0, 1.0),
    ]
    # Select the current position
    ndc_pos = positions[index]
    # Store positions and the view direction in the world
    out_pos = ndc_pos  # noqa - shader output
    v_texcoord = vec3(ndc_pos.xy * 0.5 + 0.5, 0.0)  # noqa - shader output


@python2shader
def fragment_shader_gradient(
    v_texcoord: (pyshader.RES_INPUT, 0, vec3),
    u_background: (pyshader.RES_UNIFORM, (0, 2), BackgroundMaterial.uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    f = v_texcoord.xy
    color = (
        u_background.color_bottom_left * (1.0 - f.x) * (1.0 - f.y)
        + u_background.color_bottom_right * f.x * (1.0 - f.y)
        + u_background.color_top_left * (1.0 - f.x) * f.y
        + u_background.color_top_right * f.x * f.y
    )
    out_color = color  # noqa - shader output


@python2shader
def vertex_shader_skybox(
    index: (pyshader.RES_INPUT, "VertexId", "i32"),
    out_pos: (pyshader.RES_OUTPUT, "Position", vec4),
    v_texcoord: (pyshader.RES_OUTPUT, 0, vec3),
    u_stdinfo: (pyshader.RES_UNIFORM, (0, 0), stdinfo_uniform_type),
    u_wobject: (pyshader.RES_UNIFORM, (0, 1), wobject_uniform_type),
):
    # Define positions at the four corners of the viewport, at the largest depth
    positions = [
        vec4(-1.0, -1.0, 1.0, 1.0),
        vec4(+1.0, -1.0, 1.0, 1.0),
        vec4(-1.0, +1.0, 1.0, 1.0),
        vec4(+1.0, +1.0, 1.0, 1.0),
    ]
    # Select the current position, and create another pos just behind it
    ndc_pos1 = positions[index]
    ndc_pos2 = vec4(ndc_pos1.xy, ndc_pos1.z + 0.1, ndc_pos1.z)
    # Project both points to world coordinates
    inv_proj = matrix_inverse(
        u_stdinfo.projection_transform
        * u_stdinfo.cam_transform
        * u_wobject.world_transform
    )
    wpos1 = inv_proj * ndc_pos1
    wpos2 = inv_proj * ndc_pos2
    wpos1 = wpos1.xyzw / wpos1.w
    wpos2 = wpos2.xyzw / wpos2.w
    # Store positions and the view direction in the world
    out_pos = ndc_pos1  # noqa - shader output
    v_texcoord = wpos2.xyz - wpos1.xyz  # noqa - shader output


@python2shader
def fragment_shader_tex_rgba(
    v_texcoord: (pyshader.RES_INPUT, 0, vec3),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "undecided"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord.xyz))
    out_color = color / 255.0  # noqa - shader output


@python2shader
def fragment_shader_tex_gray(
    v_texcoord: (pyshader.RES_INPUT, 0, vec3),
    s_sam: (pyshader.RES_SAMPLER, (1, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (1, 1), "undecided"),
    out_color: (pyshader.RES_OUTPUT, 0, vec4),
):
    color = vec4(t_tex.sample(s_sam, v_texcoord.xyz))
    out_color = color  # noqa - shader output


@register_wgpu_render_function(Background, BackgroundMaterial)
def background_renderer(wobject, render_info):

    material = wobject.material
    vertex_shader = vertex_shader_simple
    fragment_shader = fragment_shader_tex_rgba
    bindings0 = {
        0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo_uniform),
        1: (wgpu.BindingType.uniform_buffer, render_info.wobject_uniform),
        2: (wgpu.BindingType.uniform_buffer, material.uniform_buffer),
    }
    bindings1 = {}

    if isinstance(material, BackgroundImageMaterial) and material.map is not None:
        if isinstance(material.map, Texture):
            raise TypeError("material.map is a Texture, but must be a TextureView")
        elif not isinstance(material.map, TextureView):
            raise TypeError("material.map must be a TextureView")
        bindings1[0] = wgpu.BindingType.sampler, material.map
        bindings1[1] = wgpu.BindingType.sampled_texture, material.map
        # Select shader
        if material.map.view_dim == "cube":
            vertex_shader = vertex_shader_skybox
            tex_info = "cube "
        elif material.map.view_dim == "2d":
            vertex_shader = vertex_shader_simple
            tex_info = "2d "
        else:
            raise ValueError(
                "BackgroundImageMaterial should have map with texture view 2d or cube."
            )
        if "rgba" in material.map.format:
            fragment_shader = fragment_shader_tex_rgba
        else:
            fragment_shader = fragment_shader_tex_gray
        # Use a version of the shader for float textures if necessary
        if "float" in material.map.format:
            tex_info += "f32"
        else:
            tex_info += "i32"
        if not hasattr(fragment_shader, "shader_version_" + tex_info):
            func = fragment_shader.input
            tex_anno = func.__annotations__["t_tex"]
            func.__annotations__["t_tex"] = tex_anno[:2] + (tex_info,)
            setattr(fragment_shader, "shader_version_" + tex_info, python2shader(func))
            func.__annotations__["t_tex"] = tex_anno
        fragment_shader = getattr(fragment_shader, "shader_version_" + tex_info)
    else:
        vertex_shader = vertex_shader_simple
        fragment_shader = fragment_shader_gradient

    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.triangle_strip,
            "indices": 4,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]
