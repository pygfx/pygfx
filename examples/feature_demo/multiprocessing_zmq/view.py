"""
View data from a zmq publisher
==============================

Example that demonstrates how to use zmq to receive data from a another process and visualize it.
Use in conjunction with "compute.py" in this directory.

First run `view.py`, and then separately run `compute.py`
"""

# sphinx_gallery_pygfx_docs = 'hidden'
# sphinx_gallery_pygfx_test = 'off'

import numpy as np
import zmq

import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run

context = zmq.Context()

# create subscriber
sub = context.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b"")

# keep only the most recent message
sub.setsockopt(zmq.CONFLATE, 1)

# publisher address and port
sub.connect("tcp://127.0.0.1:5555")


def get_bytes():
    """
    Gets the bytes from the publisher
    """
    try:
        b = sub.recv(zmq.NOBLOCK)
    except zmq.Again:
        pass
    else:
        return b

    return None


canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
camera = gfx.OrthographicCamera()

# initialize some data, must be of same dtype and shape as data sent by publisher
data = np.random.rand(512, 512).astype(np.float32)

image = gfx.Image(
    geometry=gfx.Geometry(grid=gfx.Texture(data, dim=2)),
    material=gfx.ImageBasicMaterial(clim=(0, 1), map=gfx.cm.plasma),
)

scene.add(image)
camera.show_object(scene)


def update_frame():
    # receive bytes
    b = get_bytes()

    if b is not None:
        # numpy array from bytes, MUST specify dtype and make sure it matches what you sent
        a = np.frombuffer(b, dtype=np.float32).reshape(512, 512)

        # set image data
        image.geometry.grid.data[:] = a
        image.geometry.grid.update_full()

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(update_frame)
    run()
