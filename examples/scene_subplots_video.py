"""
An example combining `synced_video.py` with subplots.
Double click to re-center the images.
"""

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
cntl_defaults = list()
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
    camera.position.set(*center_cam_pos)
    cameras.append(camera)

    # create viewport for this image
    viewport = gfx.Viewport(renderer)
    viewports.append(viewport)

    # controller for pan & zoom
    controller = gfx.PanZoomController(camera.position.clone())
    controller.add_default_event_handlers(viewport, camera)
    controllers.append(controller)

    # # get the initial controller params so the camera can be reset later
    cntl_defaults.append(controller.save_state())


@renderer.add_event_handler("resize")
def layout(event=None):
    """
    Update the viewports when the canvas is resized
    """
    w, h = renderer.logical_size
    w2, h2 = w / 2, h / 2
    viewports[0].rect = 10, 10, w2, h2
    viewports[1].rect = w / 2 + 5, 10, w2, h2
    viewports[2].rect = 10, h / 2 + 5, w2, h2
    viewports[3].rect = w / 2 + 5, h / 2 + 5, w2, h2


def animate():
    for img in images:
        # create new image data
        img.geometry.grid.data[:] = np.random.rand(*dims).astype(np.float32) * 255
        img.geometry.grid.update_range((0, 0, 0), img.geometry.grid.size)

    for camera, controller in zip(cameras, controllers):
        # update with the pan & zoom params
        controller.update_camera(camera)

    # render the viewports
    for viewport, s, c in zip(viewports, scenes, cameras):
        viewport.render(s, c)

    renderer.flush()
    canvas.request_draw()


def center_objects(ev):
    for con, cam, img in zip(controllers, cameras, images):
        con.show_object(cam, img)
        con.zoom(1.3)


renderer.add_event_handler(center_objects, "double_click")


layout()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()

# Use with a Qt app or `jupyter_rfb` to utilize the `reset_camera`:
# from ipywidgets import Button
#
# reset_button = Button(description="Reset View")
#
# def on_button_clicked(b):
#     global reset_cameras
#     reset_cameras = True
#
# reset_button.on_click(on_button_clicked)
#
# reset_button
