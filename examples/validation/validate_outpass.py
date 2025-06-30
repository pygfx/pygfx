"""
Validate OutPass
================

Validate the different filters for the output pass, (i.e. flush).
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la

# Setting up the outer scene that shows the same scene multiple, with different ppaa

n = 4

canvas = RenderCanvas(size=(500 * n, 500))
renderer = gfx.renderers.WgpuRenderer(canvas)

textures = []
meshes = []


for i in range(n):
    tex = gfx.Texture(dim=2, size=(500, 500, 1), format="4xu1")
    mesh = gfx.Mesh(
        gfx.plane_geometry(),
        gfx.MeshBasicMaterial(map=gfx.TextureMap(tex, filter="nearest")),
    )
    mesh.local.position = (0.5 + i, 0.5, 0)
    textures.append(tex)
    meshes.append(mesh)

scene0 = gfx.Scene().add(*meshes)
camera0 = gfx.OrthographicCamera()
camera0.show_rect(0, n, 0, 1)


# Setup the different renderers

renderers = [
    gfx.renderers.WgpuRenderer(textures[0], pixel_ratio=0.2, pixel_filter="nearest"),
    gfx.renderers.WgpuRenderer(textures[1], pixel_ratio=0.2, pixel_filter="disk"),
    gfx.renderers.WgpuRenderer(textures[2], pixel_ratio=0.2, pixel_filter="bspline"),
    gfx.renderers.WgpuRenderer(textures[3], pixel_ratio=0.2, pixel_filter="mitchell"),
]


# Setup the actual scene that gets rendered multiple times

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

points = gfx.Points(
    gfx.Geometry(positions=[[0, 0, -1]]),
    gfx.PointsMaterial(color="#fff", size=95, aa=False, size_space="model"),
)
scene.add(points)

mesh1 = gfx.Mesh(
    gfx.plane_geometry(50, 2),
    gfx.MeshBasicMaterial(color="#00a"),
)
mesh1.local.position = 0, 10, 0
mesh1.local.rotation = la.quat_from_vecs((0, 100, 0), (4, 100, 0))
scene.add(mesh1)

mesh2 = gfx.Mesh(
    gfx.plane_geometry(50, 2),
    gfx.MeshBasicMaterial(color="#00a"),
)
mesh2.local.position = 10, 20, 0
mesh2.local.rotation = la.quat_from_vecs((0, 100, 0), (100, 100, 0))
scene.add(mesh2)

mesh3 = gfx.Mesh(
    gfx.plane_geometry(50, 2),
    gfx.MeshBasicMaterial(color="#00a"),
)
mesh3.local.position = -20, 20, 0
mesh3.local.rotation = la.quat_from_vecs((0, 100, 0), (-200, 100, 0))
scene.add(mesh3)

positions = [[-30, -32, 0], [30, -30, 0], [0, 0, 0], [-15, -30, 0]]
line = gfx.Line(
    gfx.Geometry(positions=positions),
    gfx.LineMaterial(color="#000", thickness=2, aa=False),
)
scene.add(line)


text = gfx.Text(
    material=gfx.TextMaterial(
        color="#080",
        aa=True,
    ),
    text="Hello",
    font_size=10,
    screen_space=False,
)
text.local.position = -25, -15, 0
scene.add(text)

camera = gfx.OrthographicCamera()
camera.show_rect(-50, 50, -50, 50)

controller = gfx.PanZoomController(camera, register_events=renderer)


@canvas.request_draw
def animate():
    # Render the scene with different renderers
    for sub_renderer in renderers:
        sub_renderer.render(scene, camera)
    # Render resulting textures to the canvas
    renderer.render(scene0, camera0)


if __name__ == "__main__":
    print(__doc__)
    loop.run()
