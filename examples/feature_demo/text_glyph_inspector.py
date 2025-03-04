"""
A Tool to Pixel Peep On Text Glyphs
================================================

Run by providing it a short text. And you can zoom into your hearts content
on the text glyphs.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import argparse
import sys
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.imageshader import ImageShader

# Add argparse to take in the input text to inspect

parser = argparse.ArgumentParser(description="Inspect the glyphs of a font")
parser.add_argument("text", type=str, help="The text to inspect")
args = parser.parse_args()

text_to_display = args.text

canvas_size = 800, 600

canvas = WgpuCanvas(size=canvas_size)
renderer = gfx.renderers.WgpuRenderer(canvas)

w, h = canvas.get_logical_size()

viewport_atlas = gfx.Viewport(renderer, rect=(0, 0, w // 2, h))
viewport_text = gfx.Viewport(renderer, rect=(w // 2, 0, w - w // 2, h))

atlas_scene = gfx.Scene()
atlas_scene.add(gfx.Background.from_color("#888"))
atlas_camera = gfx.OrthographicCamera(100, 100)
atlas_camera.local.scale_y = -1

glyph_atlas = gfx.utils.text.glyph_atlas

atlas_viewer = gfx.Mesh(
    gfx.plane_geometry(100, 100),
    gfx.MeshBasicMaterial(color="red"),
)
atlas_scene.add(atlas_viewer)

atlas_controller = gfx.PanZoomController(atlas_camera, register_events=viewport_atlas)

text_scene = gfx.Scene()
text_scene.add(gfx.Background.from_color("#888"))
text_camera = gfx.OrthographicCamera(100, 100)

text_material = gfx.TextMaterial(
    color="#06E", outline_color="#000", outline_thickness=0.2
)
text_geometry = gfx.TextGeometry(text_to_display, font_size=18, screen_space=False)
text_object = gfx.Text(text_geometry, text_material)

text_scene.add(text_object)

text_controller = gfx.PanZoomController(text_camera, register_events=viewport_text)

gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_imgui():
    global text_to_display
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)
    # I can't get line inputs to work....
    changed, text = imgui.input_text("Characters to Display", text_to_display, 100)
    if changed:
        print(text)
        text_object.text = text
        text_to_display = text
    # Creata float slider for the font size
    imgui.text("Font size:")
    changed, font_size = imgui.slider_float(
        "##font_size", text_geometry.font_size, 1, 100
    )
    if changed:
        text_geometry.font_size = font_size

    changed, outline_thickness = imgui.slider_float(
        "Outline Thickness", text_material.outline_thickness, 0, 1.0
    )
    if changed:
        text_material.outline_thickness = outline_thickness

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


def animate():
    w, h = canvas.get_logical_size()

    viewport_atlas.rect = (0, 0, w // 2, h)
    viewport_text.rect = (w // 2, 0, w - w // 2, h)

    viewport_atlas.render(atlas_scene, atlas_camera)
    viewport_text.render(text_scene, text_camera)

    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


gui_renderer.set_gui(draw_imgui)
if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
