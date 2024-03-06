"""
Compute data and send over zmq to be rendered by pygfx in another process
=========================================================================

Example that demonstrates how to use zmq send data to another process that then uses pygfx to visualize it.
Use in conjunction with "view.py" in this directory.

First run `view.py`, and then separately run `compute.py`
"""

# sphinx_gallery_pygfx_render = False

import numpy as np
import zmq

context = zmq.Context()

# create publisher
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

for i in range(5_000):
    # make some data, make note of the dtype
    data = np.random.rand(512, 512).astype(np.float32)

    # sent bytes over the socket
    socket.send(data.tobytes())
