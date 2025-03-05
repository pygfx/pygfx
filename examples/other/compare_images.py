"""
A Tool to Compare Two Images of Identical Shapes
================================================

While mostly an internal tool, it does show how one can combine multiple scenes,
all sharing the same data to display it different ways.

It shows a grid with 4 images:
- Top Left: Reference image
- Top Right: Image to compare
- Bottom Left: Difference between the images (RGB)
- Bottom Right: Difference between the images (Alpha)

One can pan and zoom, as well as adjust the contrast of the images.

This example can be used as an example of how a custom shader can be created
in order to discard pixels based on their color.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import argparse
import sys
import os
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader

if "PYTEST_CURRENT_TEST" not in os.environ:
    parser = argparse.ArgumentParser(description="Image Comparison")
    # Require two optional arguements
    parser.add_argument("images", nargs="*", help="The two images to compare")
    args = parser.parse_args()


if "PYTEST_CURRENT_TEST" in os.environ or len(args.images) == 0:
    # Cannot use __file__ since sphinx-gallery does not support it
    this_file = sys.argv[0]
    gfx_examples_screenshot_dir = (
        Path(this_file).parent.parent / "screenshots"
    ).resolve()
    reference = imageio.imread(
        gfx_examples_screenshot_dir / "validate_blend_weightedplus.png"
    )
    image = imageio.imread(gfx_examples_screenshot_dir / "validate_blend_dither.png")
elif len(args.images) == 2:
    reference = imageio.imread(args.images[0])
    image = imageio.imread(args.images[1])
else:
    print("Please provide two images to compare")
    sys.exit(1)

if image.shape != reference.shape:
    print("Image shapes do not match")
    sys.exit(1)

if image.dtype != reference.dtype:
    print("Image dtypes do not match")
    sys.exit(1)

if image.dtype.name != "uint8":
    print("Only uint8 images are supported (for now)")
    sys.exit(1)

difference_rgb = np.abs(
    image[..., :3].astype(int) - reference[..., :3].astype(int)
).astype(image.dtype)

if image.shape[-1] == 4:
    difference_alpha = np.abs(
        image[..., 3].astype(int) - reference[..., 3].astype(int)
    ).astype(image.dtype)
else:
    difference_alpha = np.zeros_like(image)

canvas_size = 800, 800

canvas = WgpuCanvas(size=canvas_size)
renderer = gfx.renderers.WgpuRenderer(canvas)

w, h = canvas.get_logical_size()

viewport_reference = gfx.Viewport(renderer, rect=(0, 0, w // 2, h // 2))
viewport_image = gfx.Viewport(renderer, rect=(w // 2, 0, w - w // 2, h // 2))
viewport_diff_rgb = gfx.Viewport(renderer, rect=(0, h // 2, w // 2, h - h // 2))
viewport_diff_alpha = gfx.Viewport(
    renderer, rect=(w // 2, h // 2, w - w // 2, h - h // 2)
)

scene = gfx.Scene()
camera = gfx.OrthographicCamera(w // 2, h // 2)
camera.local.scale_y = -1

image_material = gfx.ImageBasicMaterial(clim=(0, 255))

diff_material = gfx.ImageBasicMaterial(
    clim=(0, max(difference_alpha.max(), difference_rgb.max(), 1))
)

geometry_reference = gfx.Geometry(grid=gfx.Texture(reference[..., :3], dim=2))
geometry_image = gfx.Geometry(grid=gfx.Texture(image[..., :3], dim=2))
geometry_diff_rgb = gfx.Geometry(grid=gfx.Texture(difference_rgb, dim=2))
geometry_diff_alpha = gfx.Geometry(grid=gfx.Texture(difference_alpha, dim=2))

gfx_reference = gfx.Image(geometry_reference, image_material)
gfx_image = gfx.Image(geometry_image, image_material)


# We create a dedicated pixel peeper shader which will cause "0" differences
# to be discareded
class ImageErrorOverlay(gfx.Image):
    pass


@register_wgpu_render_function(ImageErrorOverlay, gfx.ImageBasicMaterial)
class ErrorOverlayShader(ImageShader):
    def get_code(self):
        # Note that we provide no guarantees on the continuity of the shader code
        # using text manipulation like this is not guaranteed to work as part
        # of our API.
        # we would typically recommend that users build up their own objects
        # and shaders.
        return (
            super()
            .get_code()
            .replace(
                """
    let out_color = vec4<f32>(physical_color, opacity);
""",
                """
    let out_color = vec4<f32>(physical_color, opacity);
    if (length(out_color.rgb) < 1e-3) {
        discard;
    }
""",
            )
        )


gfx_reference_overlay_rgb = ImageErrorOverlay(geometry_diff_rgb, diff_material)
gfx_reference_overlay_rgb.local.z = 1
gfx_reference_overlay_rgb.visible = False

gfx_image_overlay_rgb = ImageErrorOverlay(geometry_diff_rgb, diff_material)
gfx_image_overlay_rgb.local.z = 1
gfx_image_overlay_rgb.visible = False

gfx_reference_overlay_alpha = ImageErrorOverlay(geometry_diff_alpha, diff_material)
gfx_reference_overlay_alpha.local.z = 1
gfx_reference_overlay_alpha.visible = False

gfx_image_overlay_alpha = ImageErrorOverlay(geometry_diff_alpha, diff_material)
gfx_image_overlay_alpha.local.z = 1
gfx_image_overlay_alpha.visible = False

scene_reference = gfx.Scene()
scene_reference.add(gfx.Background.from_color("#111111"))
scene_reference.add(gfx_reference)
scene_reference.add(gfx_reference_overlay_rgb)
scene_reference.add(gfx_reference_overlay_alpha)

scene_image = gfx.Scene()
scene_image.add(gfx.Background.from_color("#111111"))
scene_image.add(gfx_image)
scene_image.add(gfx_image_overlay_rgb)
scene_image.add(gfx_image_overlay_alpha)

gfx_diff_rgb = gfx.Image(geometry_diff_rgb, diff_material)
gfx_diff_alpha = gfx.Image(geometry_diff_alpha, diff_material)

scene_diff_rgb = gfx.Scene()
scene_diff_rgb.add(gfx.Background.from_color("#111111"))
scene_diff_rgb.add(gfx_diff_rgb)

scene_diff_alpha = gfx.Scene()
scene_diff_alpha.add(gfx.Background.from_color("#111111"))
scene_diff_alpha.add(gfx_diff_alpha)


camera.show_object(scene_reference)

controller_reference = gfx.PanZoomController(camera, register_events=viewport_reference)
controller_image = gfx.PanZoomController(camera, register_events=viewport_image)
controller_diff_rgb = gfx.PanZoomController(camera, register_events=viewport_diff_rgb)
controller_diff_alpha = gfx.PanZoomController(
    camera, register_events=viewport_diff_alpha
)

gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_imgui():
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)

    if is_expand:
        clim_min, clim_max = image_material.clim
        changed, clim_min, clim_max = imgui.drag_int_range2(
            "Images",
            int(clim_min),
            int(clim_max),
            1.0,
            0,
            255,
            "Min: %d",
            "Max: %d",
        )
        if changed:
            image_material.clim = clim_min, clim_max

        clim_min, clim_max = diff_material.clim
        changed, clim_min, clim_max = imgui.drag_int_range2(
            "Difference",
            int(clim_min),
            int(clim_max),
            1.0,
            0,
            255,
            "Min: %d",
            "Max: %d",
        )
        if changed:
            diff_material.clim = clim_min, clim_max

        changed, visible = imgui.checkbox(
            "Overlay Diff RGB", gfx_reference_overlay_rgb.visible
        )
        if changed:
            gfx_reference_overlay_rgb.visible = visible
            gfx_image_overlay_rgb.visible = visible

        changed, visible = imgui.checkbox(
            "Overlay Diff Alpha", gfx_reference_overlay_alpha.visible
        )
        if changed:
            gfx_reference_overlay_alpha.visible = visible
            gfx_image_overlay_alpha.visible = visible

        changed, swap_images = imgui.checkbox(
            "Swap Images", gfx_image.geometry != geometry_image
        )
        if changed:
            if swap_images:
                gfx_reference.geometry = geometry_image
                gfx_image.geometry = geometry_reference
            else:
                gfx_reference.geometry = geometry_reference
                gfx_image.geometry = geometry_image

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


def animate():
    w, h = canvas.get_logical_size()

    viewport_reference.rect = (0, 0, w // 2, h // 2)
    viewport_image.rect = (w // 2, 0, w - w // 2, h // 2)
    viewport_diff_rgb.rect = (0, h // 2, w // 2, h - h // 2)
    viewport_diff_alpha.rect = (w // 2, h // 2, w - w // 2, h - h // 2)

    viewport_reference.render(scene_reference, camera)
    viewport_image.render(scene_image, camera)
    viewport_diff_rgb.render(scene_diff_rgb, camera)
    viewport_diff_alpha.render(scene_diff_alpha, camera)

    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


gui_renderer.set_gui(draw_imgui)
if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
