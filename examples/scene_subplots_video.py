"""
An example combining `synced_video.py` with subplots
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
cmaps = \
    [
        gfx.cm.inferno,
        gfx.cm.plasma,
        gfx.cm.magma,
        gfx.cm.viridis
    ]

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
        gfx.Geometry(grid=gfx.Texture(np.random.rand(*dims).astype(np.float32) * 255, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255), map=cmaps[i])
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

    # get the initial controller params so the camera can be reset later
    cntl_default = dict()
    cntl_default['distance'] = controller.distance
    cntl_default['zoom_value'] = controller.zoom_value
    cntl_default['target'] = controller.target.clone()
    cntl_defaults.append(cntl_default)

w_div = 2
h_div = 2


# 2x2 arrangement for the viewports
def produce_rect(w, h):
    return [
        (0, 0, w / w_div, h / h_div),
        (w / w_div, 0, w / w_div, h / h_div),
        (0, h / h_div, w / w_div, h / h_div),
        (w / w_div, h / h_div, w / w_div, h / h_div)
    ]


reset_cameras = False


def animate():
    for img in images:
        # create new image data
        img.geometry.grid.data[:] = np.random.rand(*dims).astype(np.float32) * 255
        img.geometry.grid.update_range((0, 0, 0), img.geometry.grid.size)
        # img.geometry.grid = gfx.Texture(np.random.rand(*dims).astype(np.float32) * 255, dim=2)

    w, h = canvas.get_logical_size()

    rects = produce_rect(w, h)

    global reset_cameras

    # reset the cameras if `reset_camera` is set to True
    if reset_cameras:
        for camera, controller, cntrl_default in zip(cameras, controllers, cntl_defaults):
            pan_delta = cntl_default['target'].clone().sub(camera.position)  # find the dx, dy
            controller.pan(pan_delta)  # pan to initial state

            # set zoom and distance to initial state
            controller.zoom_value = cntl_default['zoom_value']
            controller.distance = cntl_default['distance']

            # update camera with the new params
            controller.update_camera(camera)

        reset_cameras = False
    else:
        for camera, controller, cntrl_default in zip(cameras, controllers, cntl_defaults):
            # if not reset, update with the pan & zoom params
            controller.update_camera(camera)

    # render the viewports
    for viewport, s, c, r in zip(viewports, scenes, cameras, rects):
        viewport.rect = r
        viewport.render(s, c)

    renderer.flush()
    canvas.request_draw()


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


