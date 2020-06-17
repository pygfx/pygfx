import wgpu  # only for flags/enums
from pyshader import python2shader
from pyshader import f32, vec2, vec4

from . import register_wgpu_render_function, stdinfo_uniform_type
from ...objects import Points
from ...materials import PointsMaterial, GaussianPointsMaterial


@python2shader
def vertex_shader(
    in_pos: ("input", 0, vec4),
    u_stdinfo: ("uniform", (0, 0), stdinfo_uniform_type),
    u_points: ("uniform", (1, 0), PointsMaterial.uniform_type),
    out_pos: ("output", "Position", vec4),
    out_point_size: ("output", "PointSize", f32),
    out_size: ("output", 0, f32),
):
    world_pos = u_stdinfo.world_transform * vec4(in_pos.xyz, 1.0)
    ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos

    scale_factor = u_stdinfo.physical_size.x / u_stdinfo.logical_size.x
    size = u_points.size * scale_factor + 1.5  # plus some for aa

    out_pos = ndc_pos  # noqa - shader output
    out_point_size = size  # noqa - shader output
    out_size = size  # noqa - shader output


@python2shader
def fragment_shader_points(
    in_size: ("input", 0, f32),
    in_point_coord: ("input", "PointCoord", vec2),
    u_points: ("uniform", (1, 0), PointsMaterial.uniform_type),
    out_color: ("output", 0, vec4),
):
    # See https://github.com/vispy/vispy/blob/master/vispy/visuals/markers.py
    color = u_points.color
    pcoord_pixels = in_point_coord * in_size
    hsize = 0.5 * (in_size - 1.5)
    aa_width = 1.0
    d = distance(pcoord_pixels, vec2(hsize, hsize))
    if d <= hsize - 0.5 * aa_width:
        out_color = u_points.color.rgba  # noqa - shader output
    elif d <= hsize + 0.5 * aa_width:
        alpha = 0.5 + (hsize - d) / aa_width
        alpha = alpha ** 2  # this works better
        out_color = vec4(color.rgb, color.a * alpha)  # noqa - shader output
    else:
        return  # discard


@python2shader
def fragment_shader_gaussian(
    in_size: ("input", 0, f32),
    in_point_coord: ("input", "PointCoord", vec2),
    u_points: ("uniform", (1, 0), PointsMaterial.uniform_type),
    out_color: ("output", 0, vec4),
):
    color = u_points.color
    pcoord_pixels = in_point_coord * in_size
    hsize = 0.5 * (in_size - 1.5)
    sigma = hsize / 3.0
    d = distance(pcoord_pixels, vec2(hsize, hsize))
    if d <= hsize:
        t = d / sigma
        a = exp(-0.5 * t * t)
        out_color = vec4(color.rgb, color.a * a)  # noqa - shader output
    else:
        return  # discard


@register_wgpu_render_function(Points, PointsMaterial)
def points_renderer(wobject, render_info):
    """ Render function capable of rendering meshes displaying a volume slice.
    """

    geometry = wobject.geometry
    material = wobject.material

    # Collect vertex buffers
    n = geometry.positions.nitems
    vertex_buffers = [geometry.positions]

    # Collect bindings
    bindings0 = {0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo)}
    bindings1 = {0: (wgpu.BindingType.uniform_buffer, material.uniform_buffer)}

    if isinstance(material, GaussianPointsMaterial):
        fragment_shader = fragment_shader_gaussian
    else:
        fragment_shader = fragment_shader_points

    # Put it together!
    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": wgpu.PrimitiveTopology.point_list,
            "indices": (range(n), range(1)),
            "vertex_buffers": vertex_buffers,
            "bindings0": bindings0,
            "bindings1": bindings1,
        }
    ]
