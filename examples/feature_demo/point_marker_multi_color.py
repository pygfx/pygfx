"""
Point Markers with Multiple Colors
==================================

We demonstrate how one can use custom signed distance functions (SDFs) to
create point markers with multiple colors. In this example we recreate an
approximation of the Pygfx logo using two SDFs, one for the outer yellow
triangle, and other for the inner blue highlight. Combining multiple colors in
a single point marker is achieved by adding two separate markers to the scene,
one for each color.

Transparent point materials are used in the scene and can be navigated with the
fly controller. The solid logo is then overlayed ontop of the scene and does
not move with the controller.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


custom_sdf = """
    let m_sqrt_3 = 1.7320508075688772;

    // coords uses WGSL coordinates.
    // we shift it so that the center of the triangle is at the origin.
    // for ease of calculations.
    var coord_for_sdf = coord / size + vec2<f32>(0.5, -0.5);

    // https://math.stackexchange.com/a/4073070
    // equilateral triangle has length of size
    //    sqrt(3) - 1
    let triangle_x = m_sqrt_3 - 1.;
    let one_minus_triangle_x = 2. - m_sqrt_3;
    let triangle_length = SQRT_2 * triangle_x;

    let pygfx_width = 0.10;

    let v1 = normalize(vec2<f32>(one_minus_triangle_x, 1));
    let r1_out = dot(coord_for_sdf, v1);
    let r1_in  = r1_out + pygfx_width;

    let v2 = normalize(vec2<f32>(-1, -one_minus_triangle_x));
    let r2_out = dot(coord_for_sdf, v2);
    let r2_in  = r2_out + pygfx_width;

    let v3 = normalize(vec2<f32>(triangle_x, -triangle_x));
    let r3_out = dot(coord_for_sdf - vec2(1, -one_minus_triangle_x), v3);
    let r3_in  = r3_out + pygfx_width;
"""

inner_triangle_sdf = (
    custom_sdf
    + """
    let inner_offset = 0.5 * (triangle_length - pygfx_width / 2.);
    let r1_out_blue = -r1_out - inner_offset;
    let r1_in_blue = r1_out_blue + pygfx_width;
    let r1_blue = max(
        max(r2_out, r3_in),
        max(r1_out_blue, -r1_in_blue)
    );

    let r2_out_blue = -r2_out - inner_offset;
    let r2_in_blue = r2_out_blue + pygfx_width;
    let r2_blue = max(
        max(r3_out, r1_in),
        max(r2_out_blue, -r2_in_blue)
    );

    let r3_out_blue = -r3_out - inner_offset;
    let r3_in_blue = r3_out_blue + pygfx_width;
    let r3_blue = max(
        max(r1_out, r2_in),
        max(r3_out_blue, -r3_in_blue)
    );

    return min(r1_blue, min(r2_blue, r3_blue)) * size;
"""
)

outer_triangle_sdf = (
    custom_sdf
    + """
    return max(
        max(r1_out, max(r2_out, r3_out)),
        -max(r1_in, max(r2_in, r3_in))
    ) * size;

"""
)


def make_logo_scene_camera(size=100, edge_width=3, padding=(5, 5)):
    geometry = gfx.Geometry(positions=[[size / 2, size / 2, 0]])
    logo_inner = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=size,
            color="#387EB9",
            marker="custom",
            custom_sdf=inner_triangle_sdf,
            edge_color="#000",
            edge_width=edge_width,
        ),
    )
    logo_outer = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=size,
            color="#FFE64B",
            marker="custom",
            custom_sdf=outer_triangle_sdf,
            edge_color="#000",
            edge_width=edge_width,
        ),
        # ensure that the outer logo doesn't go over the inner one
        # to create the desired visual effect
        render_order=logo_inner.render_order + 1,
    )
    logo = gfx.Group()
    logo.add(logo_outer, logo_inner)
    logo.local.position = (padding[0], padding[1], 0)

    camera = gfx.ScreenCoordsCamera()
    return logo, camera


# Make singleton for fast access
logo, logo_camera = make_logo_scene_camera()

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=100,
        minor_step=10,
        thickness_space="world",
        major_thickness=2,
        minor_thickness=0.1,
        infinite=True,
    ),
    orientation="xz",
)
grid.local.y = -120
scene.add(grid)

# Create a bunch of logos
n = 100
positions = np.random.normal(0, 50, (n, 3)).astype(np.float32)
sizes = np.random.rand(n).astype(np.float32) * 50
colors_inner = np.random.rand(n, 4).astype(np.float32)
colors_outer = np.random.rand(n, 4).astype(np.float32)
geometry_inner = gfx.Geometry(
    positions=positions,
    sizes=sizes,
    colors=colors_inner,
)
geometry_outer = gfx.Geometry(
    positions=positions,
    sizes=sizes,
    colors=colors_outer,
)

points_inner = gfx.Points(
    geometry_inner,
    gfx.PointsMarkerMaterial(
        marker="custom",
        custom_sdf=inner_triangle_sdf,
        color_mode="vertex",
        size_mode="vertex",
        size_space="world",
    ),
)
points_outer = gfx.Points(
    geometry_outer,
    gfx.PointsMarkerMaterial(
        marker="custom",
        custom_sdf=outer_triangle_sdf,
        color_mode="vertex",
        size_mode="vertex",
        size_space="world",
    ),
)
scene.add(points_inner, points_outer)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene)
controller = gfx.FlyController(camera, register_events=renderer)


def render_full_scene():
    renderer.render(scene, camera, flush=False)
    renderer.render(logo, logo_camera)


if __name__ == "__main__":
    canvas.request_draw(render_full_scene)
    run()
