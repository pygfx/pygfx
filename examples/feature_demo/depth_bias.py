"""
Depth bias
==========

Two tilted planes, each with a smaller square lying exactly on it. On the
left plane the square z-fights with the plane and is partly hidden. The
right plane is pushed back in the depth test with ``depth_bias`` and
``depth_bias_slope_scale``, so the square on it is whole.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(500, 300))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()


def add_plane_with_square(x, depth_bias, depth_bias_slope_scale):
    plane = gfx.Mesh(
        gfx.plane_geometry(2, 2),
        gfx.MeshBasicMaterial(
            color="#ddd",
            depth_bias=depth_bias,
            depth_bias_slope_scale=depth_bias_slope_scale,
        ),
    )
    square = gfx.Mesh(
        gfx.plane_geometry(1.2, 1.2), gfx.MeshBasicMaterial(color="#336699")
    )
    group = gfx.Group()
    group.add(plane, square)
    group.local.x = x
    group.local.euler_x = 1.0  # tilt, so the planes have a depth slope
    scene.add(group)


add_plane_with_square(-1.2, 0, 0.0)
add_plane_with_square(+1.2, 2, 1.0)

camera = gfx.OrthographicCamera(5, 3)
controller = gfx.OrbitController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
