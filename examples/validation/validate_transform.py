"""
Validate transforms
===================

Apply a variety of transform combinations.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

scene = gfx.Scene()

geometry = gfx.box_geometry(1, 1, 1)
material_a = gfx.MeshBasicMaterial(
    color=(0, 0.5, 0.5), wireframe=True, wireframe_thickness=3
)
material_b = gfx.MeshBasicMaterial(color=(0, 1, 1))


def create_boxes(ttype1, value1, ttype2, value2, ttype3, value3, *, state_basis):
    box_a = gfx.Mesh(geometry, material_a)
    box_b = gfx.Group()
    box_c = gfx.Mesh(geometry, material_b)

    for box in (box_a, box_b, box_c):
        box.local.state_basis = state_basis

    setattr(box_a.local, ttype1, value1)
    setattr(box_b.local, ttype2, value2)
    setattr(box_c.local, ttype3, value3)

    scene.add(box_a.add(box_b.add(box_c)))


for i, state_basis in enumerate(["components", "matrix"]):
    dy = i * 4
    for transforms in [
        ("position", (0, dy, 0), "position", (0, 1, 0), "scale", (1.0, 1.5, 1.0)),
        ("position", (2, dy, 0), "scale", (1.0, 1.5, 1.0), "position", (0, 1, 0)),
        ("position", (4, dy, 0), "position", (0, 1, 0), "euler_z", 0.5),
        ("position", (6, dy, 0), "euler_z", 0.5, "position", (0, 1, 0)),
        ("position", (8, dy, 0), "euler_z", 0.5, "scale", (1.0, 1.5, 1.0)),
        ("position", (10, dy, 0), "scale", (1.0, 1.5, 1.0), "euler_z", 0.5),
    ]:
        create_boxes(*transforms, state_basis=state_basis)


camera = gfx.OrthographicCamera(5, 5)
camera.show_rect(-2, 12, -2, 6)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
