"""
Show an image and print the x, y image data coordinates for click events.
"""

import imageio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# %% add image

im = imageio.imread("imageio:astronaut.png")

image = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(im, dim=2)),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
scene.add(image)

camera = gfx.OrthographicCamera(512, 512)
camera.position.set(256, 256, 0)
camera.scale.y = -1

xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]

colors = np.vstack([[0.0, 0.0, 1.0, 1.0]] * len(xx)).astype(np.float32)

points = gfx.Points(
    gfx.Geometry(positions=[(x, y, 0) for x, y in zip(xx, yy)], colors=colors),
    gfx.PointsMaterial(vertex_colors=True, size=10),
)
points.position.z = 1  # move points in front of the image
scene.add(points)


previous_selection = None


def event_handler(event):
    print(
        f"Canvas click coordinates: {event.x, event.y}\n"
        f"Click position in coordinate system of image, i.e. data coordinates of click event: {event.pick_info['index']}\n"
        f"Other `pick_info`: {event.pick_info}"
    )

    global previous_selection

    if previous_selection is not None:
        # reset colors to blue
        blues = np.vstack([[0.0, 0.0, 1.0, 1.0]] * 3).astype(np.float32)
        points.geometry.colors.data[previous_selection] = blues

        for idx in previous_selection.tolist():
            points.geometry.colors.update_range(idx, size=1)

    # set the color of the 3 closest points to red
    positions = points.geometry.positions.data
    event_position = np.array([*event.pick_info["index"], 0])
    closest = np.linalg.norm(positions - event_position, axis=1).argsort()
    points.geometry.colors.data[closest[:3]] = np.array([1.0, 0.0, 0.0, 1.0])
    for idx in closest[:3].tolist():
        # only mark the changed points for synchronization to the GPU
        points.geometry.colors.update_range(idx, size=1)

    previous_selection = closest[:3]

    renderer.request_draw()


image.add_event_handler(event_handler, "click")


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
