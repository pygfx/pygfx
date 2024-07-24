"""
Points with different markers
=============================

* All available marker shapes are shown.
* Shows red, green and blue faces. Then a semi-transparent face, and finally a fully-transparent face.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 1000))
renderer = gfx.WgpuRenderer(canvas)

colors = np.array(
    [
        [1.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 0.5, 1.0],
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.5, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ],
    np.float32,
)

npoints = len(colors)

positions = np.zeros((npoints, 3), np.float32)
positions[:, 0] = np.arange(npoints) * 2
geometry = gfx.Geometry(positions=positions, colors=colors)


scene = gfx.Scene()
scene.add(gfx.Background.from_color("#bbb", "#777"))

pygfx_sdf = """
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

    let inner_triangle = min(r1_blue, min(r2_blue, r3_blue));

    let outer_triangle = max(
        max(r1_out, max(r2_out, r3_out)),
        -max(r1_in, max(r2_in, r3_in))
    );

    return min(inner_triangle, outer_triangle) * size;
"""


y = 0
for marker in gfx.MarkerShape:
    y += 2
    line = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=30,
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=3 if not marker == "custom" else 1,
            custom_sdf=pygfx_sdf if marker == "custom" else None,
        ),
    )
    line.local.y = -y
    line.local.x = 1
    scene.add(line)

    text = gfx.Text(
        gfx.TextGeometry(
            marker, anchor="middle-right", font_size=20, screen_space=True
        ),
        gfx.TextMaterial("#000"),
    )
    text.local.y = -y
    text.local.x = 0
    scene.add(text)

camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.7)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
