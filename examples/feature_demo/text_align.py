"""
Text justification
==================

Example demonstrating the capabilities of text to be aligned and justified
according to the user's decision.

This demo enables one to interactively control the alignment and the
justification of text anchored to the center of the screen.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'


import sys

from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx

scene = gfx.Scene()


scene.add(gfx.Background(None, gfx.BackgroundMaterial("#fff", "#000")))
if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word
        "last line"
    )

print(f"========= Text =========\n{text}\n========================")

text = gfx.Text(
    gfx.TextGeometry(
        text=text,
        font_size=40,
        screen_space=True,
        text_align="center",
        anchor="middle-center",
    ),
    gfx.TextMaterial(color="#B4F8C8", outline_color="#000", outline_thickness=0.15),
)
text.local.position = (0, 0, 0)

points = gfx.Points(
    gfx.Geometry(
        positions=[
            text.local.position,
        ],
    ),
    gfx.PointsMaterial(color="#f00", size=10),
)

scene.add(text, points)

camera = gfx.OrthographicCamera(4, 3)


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))


@renderer.add_event_handler("key_down")
def change_justify(event):
    if event.key == "q":
        text.geometry.anchor = "top-left"
    elif event.key == "w":
        text.geometry.anchor = "top-middle"
    elif event.key == "e":
        text.geometry.anchor = "top-right"
    elif event.key == "a":
        text.geometry.anchor = "middle-left"
    elif event.key == "s":
        text.geometry.anchor = "middle-middle"
    elif event.key == "d":
        text.geometry.anchor = "middle-right"
    elif event.key == "z":
        text.geometry.anchor = "bottom-left"
    elif event.key == "x":
        text.geometry.anchor = "bottom-middle"
    elif event.key == "c":
        text.geometry.anchor = "bottom-right"
    elif event.key == "u":
        text.geometry.text_align = "left"
    elif event.key == "i":
        text.geometry.text_align = "center"
    elif event.key == "o":
        text.geometry.text_align = "right"
    elif event.key == "j":
        text.geometry.text_align = "justify"
    elif event.key == "h":
        text.geometry.text_align = "justify-all"
    elif event.key == "k":
        text.geometry.text_align_last = "auto"
    elif event.key == "l":
        text.geometry.text_align_last = "justify"
    elif event.key == "n":
        text.geometry.text_align_last = "left"
    elif event.key == "m":
        text.geometry.text_align_last = "center"
    elif event.key == ",":
        text.geometry.text_align_last = "right"
    elif event.key == "f":
        text.geometry.font_size *= 1.1
    elif event.key == "g":
        text.geometry.font_size /= 1.1

    print(f"Anchor: {text.geometry.anchor}")
    print(f"Text align: {text.geometry.text_align}")
    print(f"Text align last: {text.geometry.text_align_last}")
    print(f"Font size: {text.geometry.font_size}")

    renderer.request_draw()


renderer.request_draw(lambda: renderer.render(scene, camera))

print(
    """Use the keys

 q  w  e
 a  s  d
 z  x  c

To change the anchor of the text.

Use the keys

 u  i  o

to set the alignment of the text to left, middle, right respectively.

Use j to set the alignment to justify.
Use h to set the alignment to justify-all.

Use the keys

 n  m  ,

to set the alignment of the last line to left, middle, right respectively.

Use

* k to set the alignment of the last line to auto.
* l to set the alignment of the last line to justify.
* f to increase the font size.
* g to decrease the font size.

"""
)

if __name__ == "__main__":
    run()
