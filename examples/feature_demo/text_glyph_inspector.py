"""
A Tool to Pixel Peep On Text Glyphs
================================================

Run by providing it a short text. And you can zoom into your hearts content
on the text glyphs.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import argparse

from imgui_bundle import imgui
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer

import pygfx as gfx

# Add argparse to take in the input text to inspect

parser = argparse.ArgumentParser(description="Inspect the glyphs of a font")
parser.add_argument("text", nargs="*", type=str, help="The text to inspect")
args = parser.parse_args()

text_to_display = " ".join(args.text)
if not text_to_display:
    text_to_display = "Hello World"


canvas_size = 800, 800

canvas = WgpuCanvas(size=canvas_size)
renderer = gfx.renderers.WgpuRenderer(canvas)

w, h = canvas.get_logical_size()

viewport_atlas = gfx.Viewport(renderer, rect=(0, 0, w // 2, h))
viewport_text = gfx.Viewport(renderer, rect=(w // 2, 0, w - w // 2, h))

# Create the text first so that the atlas glyphs are generated
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

atlas_scene = gfx.Scene()
atlas_scene.add(gfx.Background.from_color("#888"))
atlas_camera = gfx.OrthographicCamera(100, 100)

glyph_atlas = gfx.utils.text.glyph_atlas

atlas_viewer = gfx.Mesh(
    gfx.plane_geometry(100, 100),
    gfx.MeshBasicMaterial(color="red"),
)
atlas_scene.add(atlas_viewer)

atlas_controller = gfx.PanZoomController(atlas_camera, register_events=viewport_atlas)

gui_renderer = ImguiRenderer(renderer.device, canvas)


def draw_imgui():
    global text_to_display
    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)
    # I can't get line inputs to work....
    changed, text = imgui.input_text("Text", text_to_display, 100)
    if changed:
        print(text)
        text_object.text = text
        text_to_display = text

    changed, outline_thickness = imgui.slider_float(
        "Outline", text_material.outline_thickness, 0, 0.5
    )
    if changed:
        text_material.outline_thickness = outline_thickness

    # Add a checkbox for debug mode
    changed, debug_mode = imgui.checkbox(
        "Debug Mode", text_material._debug_mode is not None
    )
    if changed:
        if debug_mode:
            text_material._debug_mode = "sample_raw_pixels"
        else:
            text_material._debug_mode = None

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


def animate():
    w, h = canvas.get_logical_size()

    viewport_atlas.rect = (0, 0, w // 2, h)
    viewport_text.rect = (w // 2, 0, w - w // 2, h)

    # Update the image displayed on the atlas viewer
    # TODO: how do we make this do "nearest neighbor" interpolation??
    if atlas_viewer.material.map is not glyph_atlas.texture:
        atlas_viewer.material.map = glyph_atlas.texture

    viewport_atlas.render(atlas_scene, atlas_camera)
    viewport_text.render(text_scene, text_camera)

    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


gui_renderer.set_gui(draw_imgui)
if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
