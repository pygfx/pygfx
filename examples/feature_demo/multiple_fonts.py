"""
Multiple Fonts
==============

Example demonstrating the capabilities of text to be rendered in
multiple different fonts.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

from datetime import datetime

now = datetime.now()
text = now.strftime("%H:%M:%S")

text_noto = gfx.Text(
    gfx.TextGeometry(
        text="Noto Sans: " + text,
        # family="Noto Sans",
    ),
    gfx.TextMaterial(color="#B4F8C8", outline_color="#000", outline_thickness=0.15),
)
text_noto.local.position = (0, -40, 0)

text_humor = gfx.Text(
    gfx.TextGeometry(
        text="Humor Sans: " + text,
        family="Humor Sans",
    ),
    gfx.TextMaterial(color="#B4F8C8", outline_color="#000", outline_thickness=0.15),
)
text_humor.local.position = (0, 0, 0)

text_instructions = gfx.Text(
    gfx.TextGeometry(
        text="Click to update the clock",
    ),
    gfx.TextMaterial(color="#B4F8C8", outline_color="#000", outline_thickness=0.15),
)
text_instructions.local.position = (0, 40, 0)

scene = gfx.Scene()
scene.add(text_noto, text_humor, text_instructions)
camera = gfx.OrthographicCamera(400, 300)
renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))


@renderer.add_event_handler("pointer_down")
def change_text(event):
    now = datetime.now()
    text = now.strftime("%H:%M:%S")
    text_noto.geometry.set_text("Noto Sans: " + text)
    text_humor.geometry.set_text("Humor Sans: " + text, family="Humor Sans")
    renderer.request_draw()


renderer.request_draw(lambda: renderer.render(scene, camera))
if __name__ == "__main__":
    run()
