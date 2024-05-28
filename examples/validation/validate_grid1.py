"""
Validate the grid
=================

Show a bunch of grids touching various grid-cases.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#fff"))


# An inf grid at the floor
grid0 = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=1,
        minor_step=0.1,
        thickness_space="world",
        axis_thickness=0.1,
        major_thickness=0.02,
        minor_thickness=0.002,
        infinite=True,
    ),
    orientation="xz",
)

# A grid oriented in yz, showing major and minor ticks
grid1 = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=1,
        minor_step=0.2,
        thickness_space="screen",
        major_thickness=2,
        minor_thickness=1,
        major_color="#aa0",
        minor_color="#aa0",
        infinite=False,
    ),
    orientation="yz",
)

# A grid oriented in xy, showing how small offsets (i.e. float errors are ok)
grid2 = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=(0.9999, 1.0001),
        minor_step=0.2,
        thickness_space="screen",
        major_thickness=2,
        minor_thickness=0.5,
        major_color="#a00",
        minor_color="#a00",
        infinite=False,
    ),
    orientation="xy",
)


# A grid with weird sampling
grid3 = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=0.275,
        minor_step=0.0,
        thickness_space="screen",
        major_thickness=3,
        minor_thickness=0.75,
        major_color="#0a0",
        minor_color="#0a0",
        infinite=False,
    ),
    orientation="xy",
)
grid3.local.x = 1

# Shifted and scaled
grid4 = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=0.275,
        minor_step=0.0,
        thickness_space="screen",
        major_thickness=3,
        minor_thickness=0.75,
        major_color="#00a",
        minor_color="#00a",
        infinite=False,
    ),
    orientation="xy",
)
grid4.local.x = 2
grid4.local.y = 0.2
grid4.local.scale = 0.7, 1, 2


scene.add(grid0, grid1, grid2, grid3, grid4)

# Show box at (0.5, 0.5, 0.5) to show where the positive axii are
cube = gfx.Mesh(
    gfx.sphere_geometry(0.1),
    gfx.MeshBasicMaterial(color="#000"),
)
cube.local.position = 0.5, 0.5, 0.5
scene.add(cube)

# camera = gfx.OrthographicCamera(5, 5)
camera = gfx.PerspectiveCamera()
camera.local.position = (3.75, 1.5, 1.5)
camera.show_pos((1, 1, 0))

controller = gfx.OrbitController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
