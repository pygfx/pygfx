"""
Image YUV420 with Points Overlaid
=================================

Show an image tranmitted to the GPU in YUV colorspace with points overlaid.
This show an example of how to write a shader that is able to combine data
from multiple grids into one.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# add image

image_rgb = iio.imread("imageio:astronaut.png")


def rgb_to_yuv420_limited_range(im):
    """A pure python implementation of cv2.cvtColor(im, cv2.COLOR_RGB2YUV_I420)"""
    # import cv2
    # image_yuv420 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV_I420)
    # Define conversion matrix for RGB to YUV limited range (ITU-R BT.601)
    m = np.array(
        [[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439, -0.368, -0.071]]
    )
    offset = np.array([16, 128, 128])
    im = im.astype(np.float32)
    yuv = im @ m.T + offset
    yuv = np.clip(yuv, [16, 16, 16], [235, 240, 240])

    yuv = yuv.astype(np.uint8)
    y = yuv[..., 0]
    u = yuv[::2, ::2, 1].reshape(-1, y.shape[1])
    v = yuv[::2, ::2, 2].reshape(-1, y.shape[1])
    return np.vstack([y, u, v])


# OpenCV returns a "planar format" YUV where
# the first "plane" of data "Y" is the intensity
# YUV420 subsamples the intensity and has the same "shape"
# as original image
# The U and V planes follow but are "subsampled" by 2
# in each direction.
# So they are 1/4 of the size
# Extract them, then reshape them.
height, width = image_rgb.shape[0:2]
u_height = height // 4

image_yuv420 = rgb_to_yuv420_limited_range(image_rgb)

im_y = image_yuv420[:height]
im_u = image_yuv420[height : height + u_height]
im_v = image_yuv420[height + u_height :]
# Rehsape can cause "implicit" copies of the data, so we assign the shape which causes the data to never get copied
im_u.shape = height // 2, width // 2
im_v.shape = height // 2, width // 2

image = gfx.Image(
    gfx.Geometry(
        grid=gfx.Texture(im_y, dim=2),
        grid_u=gfx.Texture(im_u, dim=2),
        grid_v=gfx.Texture(im_v, dim=2),
    ),
    gfx.ImageBasicMaterial(
        clim=(0, 255),
    ),
)
scene.add(image)

# add points
xx = [182, 180, 161, 153, 191, 237, 293, 300, 272, 267, 254]
yy = [145, 131, 112, 59, 29, 14, 48, 91, 136, 137, 172]
sizes = np.arange(1, len(xx) + 1, dtype=np.float32)

points = gfx.Points(
    gfx.Geometry(
        positions=[(x, y, 1) for x, y in zip(xx, yy)],
        sizes=sizes,
    ),
    gfx.PointsMaterial(
        color=(0, 1, 1, 1),
        size=10,
        size_space="world",
        size_mode="vertex",
    ),
)
scene.add(points)

camera = gfx.PerspectiveCamera(0)
camera.local.scale_y = -1
camera.show_rect(-10, 522, -10, 522)

controller = gfx.PanZoomController(camera, register_events=renderer)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
