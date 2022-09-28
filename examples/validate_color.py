"""
This example draws squares of reference colors. These can be compared to
similar output from e.g. Matplotlib.
"""
# test_example = true

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

colors1 = ["#ff0000", "#770000", "#00ff00", "#007700", "#0000ff", "#000077"]
colors2 = ["#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"]

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas, gamma_correction=1.0)
camera = gfx.OrthographicCamera(6, 2)
camera.position.set(2.5, 1, 0)
scene = gfx.Scene()

plane = gfx.plane_geometry()
for i, color in enumerate(colors1):
    m = gfx.Mesh(plane, gfx.MeshBasicMaterial(color=color))
    m.position.x = i
    m.position.y = 0
    scene.add(m)
for i, color in enumerate(colors2):
    m = gfx.Mesh(plane, gfx.MeshBasicMaterial(color=color))
    m.position.x = i
    m.position.y = 1
    scene.add(m)

canvas.request_draw(lambda: renderer.render(scene, camera))


# def show_mpl():
#
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Rectangle
#
#     fig, ax = plt.subplots()
#     ax.set_facecolor("k")
#     plt.xlim([0, 6])
#     plt.ylim([-1, 3])
#
#     for i, color in enumerate(colors1):
#         ax.add_patch(Rectangle((i, 0), 1, 1, facecolor=color))
#     for i, color in enumerate(colors2):
#         ax.add_patch(Rectangle((i, 1), 1, 1, facecolor=color))
#
#     fig.show()


if __name__ == "__main__":
    print(__doc__)
    # show_mpl()
    run()
