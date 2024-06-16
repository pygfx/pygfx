def make_logo_scene_camera(size=100, edge_width=3, padding=(5, 5)):
    from .. import (
        Group,
        Geometry,
        Points,
        PointsMarkerMaterial,
        ScreenCoordsCamera,
    )

    geometry = Geometry(positions=[[size / 2, size / 2, 0]])
    logo_inner = Points(
        geometry,
        PointsMarkerMaterial(
            size=size,
            color="#387EB9",
            marker="pygfx_inner",
            edge_color="#000",
            edge_width=edge_width,
        ),
    )
    logo_outer = Points(
        geometry,
        PointsMarkerMaterial(
            size=size,
            color="#FFE64B",
            marker="pygfx_outer",
            edge_color="#000",
            edge_width=edge_width,
        ),
        # ensure that the outer logo doesn't go over the inner one
        # to create the desired visual effect
        render_order=logo_inner.render_order + 1
    )
    logo = Group()
    # logo.add(logo_inner, logo_outer)
    logo.add(logo_outer, logo_inner)
    logo.local.position = (padding[0], padding[1], 0)

    camera = ScreenCoordsCamera()
    return logo, camera


# Make singleton for fast access
logo, logo_camera = make_logo_scene_camera()


def render_with_logo(renderer, scene=None, camera=None, flush=True):
    if scene is not None and camera is not None:
        renderer.render(scene, camera, flush=False)
    renderer.render(logo, logo_camera, flush=flush)
