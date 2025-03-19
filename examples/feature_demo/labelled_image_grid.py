"""
Labelled Image Grid
===================

This example demonstrates how to create a grid of images with labels that change color when hovered over.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import random

import imageio.v3 as iio
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx

random.seed(333)

image_names = [
    "astronaut.png",
    "wood.jpg",
    "bricks.jpg",
    "wikkie.png",
    "immunohistochemistry.png",
]
images = [iio.imread(f"imageio:{name}") for name in image_names]

grid_shape = (24, 16)

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()

labels_group = gfx.Group()
scene.add(labels_group)

image_array = np.empty(grid_shape, dtype=object)
label_array = np.empty(grid_shape, dtype=object)


def add_image(img, position, spacing=10):
    texture = gfx.Texture(img, dim=2)
    geometry = gfx.Geometry(grid=texture)
    material = gfx.ImageBasicMaterial(clim=(0, 255), pick_write=True)
    image = gfx.Image(geometry, material)
    scene.add(image)

    image.world.x = (img.shape[1] + spacing) * position[0]
    image.world.y = (img.shape[0] + spacing) * position[1]

    label = add_label(image.world.x, image.world.y, text=str(position))

    image_array[position] = image
    label_array[position] = label

    def on_pointer_enter(event):
        label_array[position].material.color = gfx.Color("#FFFF00")
        renderer.request_draw()

    def on_pointer_leave(event):
        label_array[position].material.color = gfx.Color("#FFFFFF")
        renderer.request_draw()

    image.add_event_handler(on_pointer_enter, "pointer_enter")
    image.add_event_handler(on_pointer_leave, "pointer_leave")


def add_label(x, y, text):
    label = gfx.Text(
        text=text,
        font_size=20,
        screen_space=True,
        anchor="top-left",
        material=gfx.TextMaterial(
            color="#FFFFFF",
            outline_color="#000000",
            outline_thickness=0.2,
        ),
    )
    labels_group.add(label)
    label.world.x = x
    label.world.y = y
    label.world.z = 1
    return label


for position in np.ndindex(grid_shape):
    img = random.choice(images)
    add_image(img, position)

camera = gfx.PerspectiveCamera(70)
camera.show_object(scene, match_aspect=True)
camera.local.scale_y = -1


def update_text_visibility():
    min_height = 100
    max_height = 3000
    camera_height = camera.world.position[2]

    labels_group.visible = min_height <= camera_height <= max_height


controller = gfx.PanZoomController(camera, register_events=renderer)


def update_scene():
    update_text_visibility()
    renderer.render(scene, camera)


canvas.request_draw(update_scene)
run()
