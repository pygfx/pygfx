"""
Subplots Video
==============

An example combining `synced_video.py` with subplots.
Double click to re-center the images.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)

dims = (512, 512)  # image dimensions
# default cam position
center_cam_pos = (256, 256, 0)

# colormaps for each of the 4 images
cmaps = [gfx.cm.inferno, gfx.cm.plasma, gfx.cm.magma, gfx.cm.viridis]

# lists of everything necessary to make this plot
scenes = list()
cameras = list()
images = list()
controllers = list()
camera_defaults = list()
viewports = list()

for i in range(4):
    # create scene for this subplot
    scene = gfx.Scene()
    scenes.append(scene)

    # create Image WorldObject
    img = gfx.Image(
        gfx.Geometry(
            grid=gfx.Texture(np.random.rand(*dims).astype(np.float32) * 255, dim=2)
        ),
        gfx.ImageBasicMaterial(clim=(0, 255), map=cmaps[i]),
    )

    # add image to list
    images.append(img)
    scene.add(img)

    # create camera, set default position, add to list
    camera = gfx.OrthographicCamera(*dims)
    camera.show_rect(0, 512, 0, 512)
    cameras.append(camera)

    # create viewport for this image
    viewport = gfx.Viewport(renderer)
    viewports.append(viewport)

    # controller for pan & zoom
    controller = gfx.PanZoomController(camera, register_events=renderer)
    controllers.append(controller)

    # get the initial controller params so the camera can be reset later
    camera_defaults.append(camera.get_state())


@renderer.add_event_handler("resize")
def layout(event=None):
    """
    Update the viewports when the canvas is resized
    """
    w, h = renderer.logical_size
    w2, h2 = w / 2 - 15, h / 2 - 15
    viewports[0].rect = 10, 10, w2, h2
    viewports[1].rect = w / 2 + 5, 10, w2, h2
    viewports[2].rect = 10, h / 2 + 5, w2, h2
    viewports[3].rect = w / 2 + 5, h / 2 + 5, w2, h2


def animate():
    for img in images:
        # create new image data
        img.geometry.grid.data[:] = np.random.rand(*dims).astype(np.float32) * 255
        img.geometry.grid.update_full()

    # render the viewports
    for viewport, s, c in zip(viewports, scenes, cameras):
        viewport.render(s, c)

    renderer.flush()
    canvas.request_draw()


def center_objects(ev):  # center objects upon double-click event
    for _con, cam, img in zip(controllers, cameras, images):
        cam.show_object(img)


renderer.add_event_handler(center_objects, "double_click")


layout()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
