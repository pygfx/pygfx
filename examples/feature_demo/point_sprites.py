"""
Simulating grass with point sprites
===================================

This example simulates a grassy plane, using a technique that early 3D
games used a lot. We create an image with green stripes. Than place a
couple thousand of these in a plane, using a Points object.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import pygfx as gfx


# Create image of 'grass' (luminance + alpha)
im = np.zeros((20, 20, 2), np.uint8)
im[:, :, 0] = 255
lengths = np.abs(np.random.normal(1, im.shape[1], (im.shape[0],)).astype(np.int32))
for i, length in enumerate(lengths):
    im[:length, i, 1] = 100 if i % 2 else 20  # alternate high/low alpha

# Create random positions
positions = np.random.uniform(0, 100, size=(2000, 3)).astype(np.float32)
positions[:, 1] = 0

# Create point sprites
grasses = gfx.Points(
    gfx.Geometry(positions=positions),
    gfx.PointsSpriteMaterial(
        color="#8F0", sprite=gfx.Texture(im, dim=2), size=3, size_space="world"
    ),
)

# Use a camera that is more inside the field, looking from above
camera = gfx.PerspectiveCamera()
camera.show_object(grasses, view_dir=(1, -1, 1), scale=0.25)


if __name__ == "__main__":
    disp = gfx.Display(camera=camera)
    disp.show(grasses)
