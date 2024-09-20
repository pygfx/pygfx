"""
Displaying an image in a scene
==============================

Displaying an image can be done in two different ways based the details
of the information you are trying to communicate.
We consider a color image with shape (height, width, 3). In PyGFX, this is
descried as a texture.

A gfx.Image will display the texture with an offset of 0.5 pixel
in each direction assuming even sampling. This is useful for displaying
images captured with a camera or other devices where the pixel center
is the important information. By default, nearest neighbor interpolation
is used to sample the texture.

A gfx.Mesh with a plane geometry can be used to display the texture with
spanning from the one edge of the geometry to an other.
The pixel corners at integer coordinates. This is useful when one wants
to stretch the provided data to fill a specific area. Be default bilinear
interpolation is used to sample the texture.

The example allows users zoom in and out of the scene to study the
differences between the results of both provided methods.

We encourage users of pygfx to study each Object, Material, Geometry and Shader
to better understand how they can adapt the provided classes to their needs.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas_size = 512 * 2 + 50, 512 + 100
canvas = WgpuCanvas(size=canvas_size, max_fps=999)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
camera = gfx.OrthographicCamera(canvas_size[0], canvas_size[1])
camera.local.y = canvas_size[1] / 2
camera.local.scale_y = -1
camera.local.x = canvas_size[0] / 2
controller = gfx.PanZoomController(camera, register_events=renderer)

im = iio.imread("imageio:astronaut.png")
image_texture = gfx.Texture(im, dim=2)
image = gfx.Image(
    gfx.Geometry(grid=image_texture), gfx.ImageBasicMaterial(clim=(0, 255))
)

image.local.x = 100
image.local.y = 50

scene.add(image)

geometry = gfx.plane_geometry(
    width=256, height=512, width_segments=1, height_segments=1
)
image_mesh = gfx.Mesh(
    geometry,
    gfx.MeshBasicMaterial(map=image_texture),
)
image_mesh.local.x = image.local.x + 50 + 512 + 256 / 2
image_mesh.local.y = image.local.y + 512 / 2
image_mesh.local.scale_y = -1

scene.add(image_mesh)

line = gfx.Line(
    gfx.Geometry(
        positions=np.array(
            [[-8000, 50, 0], [+8000, 50, 0]],
            dtype=np.float32,
        )
    ),
    gfx.LineMaterial(color="red"),
)
scene.add(line)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
