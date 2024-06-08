import numpy as np
from .. import (
    Mesh,
    Group,
    Geometry,
    MeshBasicMaterial,
    ScreenCoordsCamera,
)


def make_logo_scene_camera(scale=100):
    start_color = np.asarray([255, 234, 92, 255], "float32") / 255
    end_color = np.asarray([255, 196, 50, 255], "float32") / 255
    start_position = np.asarray((0, 0, 0), "float32")
    end_position = np.asarray((59, 59, 0), "float32")

    outer_triangle = np.asarray(
        [
            [0, 0, 0.5],
            [24.369, 93.781, 0.5],
            [93.825, 26.700, 0.5],
        ],
        dtype="float32",
    )

    inner_triangle = np.asarray(
        [
            [8.224, 8.416, 0.5],
            [27.508, 82.626, 0.5],
            [82.469, 29.544, 0.5],
        ],
        dtype="float32",
    )

    def get_colors(positions, start_position, start_color, end_position, end_color):
        delta_start = positions - start_position
        delta_total = end_position - start_position
        outer_delta = np.dot(delta_start, delta_total) / np.dot(
            delta_total, delta_total
        )
        too_low = outer_delta < 0
        too_high = outer_delta > 1

        final_colors = start_color + outer_delta[:, None] * (end_color - start_color)
        final_colors[too_low, :] = start_color
        final_colors[too_high, :] = end_color
        return final_colors

    all_triangles = np.concatenate([outer_triangle, inner_triangle])
    all_colors = get_colors(
        all_triangles,
        start_position=start_position,
        start_color=start_color,
        end_position=end_position,
        end_color=end_color,
    )

    blue_lines_positions = np.asarray(
        [
            [16.287, 39.445, 0.4],
            [45.224, 12.870, 0.4],
            [51.350, 14.613, 0.4],
            [17.773, 45.164, 0.4],
            [46.383, 19.275, 0.4],
            [52.492, 21.013, 0.4],
            [61.401, 58.016, 0.4],
            [56.412, 62.835, 0.4],
            [50.718, 60.210, 0.4],
            [12.718, 48.942, 0.4],
            [11.266, 43.356, 0.4],
            [54.840, 56.228, 0.4],
        ],
        dtype="float32",
    )

    blue_lines_start_color = np.asarray([54, 108, 153, 255], "float32") / 255
    blue_lines_position_start = np.asarray([59, 59, 0], "float32")
    blue_lines_end_color = np.asarray([56, 126, 185, 255], "float32") / 255
    blue_lines_position_end = np.asarray([30, 30, 0], "float32")

    blue_colors = get_colors(
        blue_lines_positions,
        start_position=blue_lines_position_start,
        start_color=blue_lines_start_color,
        end_position=blue_lines_position_end,
        end_color=blue_lines_end_color,
    )

    blue_lines = Mesh(
        Geometry(
            indices=[
                (0, 1, 2),
                (2, 3, 0),
                (4, 5, 6),
                (6, 7, 4),
                (8, 9, 10),
                (10, 11, 8),
            ],
            positions=blue_lines_positions,
            colors=blue_colors,
        ),
        MeshBasicMaterial(color_mode="vertex"),
    )

    triangle = Mesh(
        Geometry(
            indices=[
                (0, 3, 1),
                (3, 4, 1),
                (1, 4, 2),
                (4, 5, 2),
                (2, 5, 0),
                (5, 3, 0),
            ],
            positions=all_triangles,
            colors=all_colors,
        ),
        MeshBasicMaterial(color_mode="vertex"),
    )
    # Seems like the coordinates were about 100 in size
    triangle.local.scale = 1 / 100, 1 / 100, 1
    blue_lines.local.scale = 1 / 100, 1 / 100, 1

    logo = Group()
    logo.add(triangle, blue_lines)
    logo.local.scale = scale, -scale, 1
    logo.local.position = (scale / 20, scale, 0)

    camera = ScreenCoordsCamera()
    return logo, camera

# Make singleton for fast access
logo, logo_camera = make_logo_scene_camera()

def render_with_logo(renderer, scene, camera, flush=True):
    renderer.render(scene, camera, flush=False)
    renderer.render(logo, logo_camera, flush=flush)
