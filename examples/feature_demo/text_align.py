"""
Text align
==========

Example demonstrating the capabilities of text to be aligned and justified
according to the user's decision.

This demo enables one to interactively control the alignment and the
justification of text anchored to the center of the screen.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import os

from wgpu.gui.auto import WgpuCanvas, run
from pathlib import Path
import pygfx as gfx

font_file = Path(__file__).parent / "SourceSans3-Regular.ttf"
if font_file.exists():
    gfx.font_manager.add_font_file(str(font_file))

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#fff", "#000"))

if "PYTEST_CURRENT_TEST" not in os.environ:
    import argparse

    parser = argparse.ArgumentParser(description="Text Alignment Demo")
    parser.add_argument(
        "--direction", type=str, default="ltr", help="Direction parameter"
    )
    parser.add_argument(
        '--ref-glyph-size', type=int, default=48, help='Reference size for the font'
    )
    parser.add_argument(
        '--family', type=str, default="Noto Sans", help='Font family'
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default=(
            "Lorem ipsum\n"
            "Bonjour World Olá\n"  # some text that isn't equal in line
            "pygfx\n"  # a line with exactly 1 word
            "last lyne"
        ),
        help="Text to display",
    )
    parser.add_argument(
        '--outline-thickness', type=float, default=0., help='Outline thickness'
    )
    parser.add_argument(
        '--inner-outline-thickness', type=float, default=0., help='Inner outline thickness'
    )
    parser.add_argument(
        '--double-shot', type=bool, default=False, help='Double shot rendering of the text'
    )
    args = parser.parse_args()
    direction = args.direction
    text = args.text
    ref_glyph_size = args.ref_glyph_size
    family = args.family
    outline_thickness = args.outline_thickness
    inner_outline_thickness = args.inner_outline_thickness
    double_shot = args.double_shot

else:
    direction = "ltr"
    text = (
        "Lorem ipsum\n"
        "Bonjour World Olá\n"  # some text that isn't equal in line
        "pygfx\n"  # a line with exactly 1 word
        "last line"
    )
    ref_glyph_size = 48
    family = "Noto Sans"
    outline_thickness = 0.
    inner_outline_thickness = 0.
    double_shot = False

print(f"========= Text =========\n{text}\n========================")
text_main = gfx.Text(
    gfx.TextGeometry(
        text=text,
        font_size=20,
        screen_space=True,
        text_align="center",
        anchor="middle-center",
        direction=direction,
        family=family,
        ref_glyph_size=ref_glyph_size,
    ),
    gfx.TextMaterial(
        color="#B4F8C8",
        outline_color="#00000000" if double_shot else "#000000FF",
        outline_thickness=outline_thickness,
        inner_outline_thickness=inner_outline_thickness,
    ),
)
text_main.local.position = (0, 0, 0)

text_outline = gfx.Text(
    gfx.TextGeometry(
        text=text,
        font_size=20,
        screen_space=True,
        text_align="center",
        anchor="middle-center",
        direction=direction,
        family=family,
        ref_glyph_size=ref_glyph_size,
    ),
    gfx.TextMaterial(
        color="#00000000",
        outline_color="#000000FF" if double_shot else "#00000000",
        outline_thickness=outline_thickness,
        inner_outline_thickness=inner_outline_thickness,
    ),
)
# Place the outline below the main text
text_outline.local.position = (0, 0, -1)

points = gfx.Points(
    gfx.Geometry(
        positions=[
            text_main.local.position,
        ],
    ),
    gfx.PointsMaterial(color="#f00", size=10),
)

scene.add(text_main, text_outline, points)

camera = gfx.OrthographicCamera(4, 3)


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))

def handle_event(event, text):
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
    elif event.key == "Z":
        text.geometry.anchor = "baseline-left"
    elif event.key == "X":
        text.geometry.anchor = "baseline-middle"
    elif event.key == "C":
        text.geometry.anchor = "baseline-right"
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

@renderer.add_event_handler("key_down")
def change_justify(event):
    handle_event(event, text_main)
    handle_event(event, text_outline)

    print(f"Anchor: {text_main.geometry.anchor}")
    print(f"Text align: {text_main.geometry.text_align}")
    print(f"Text align last: {text_main.geometry.text_align_last}")
    print(f"Font size: {text_main.geometry.font_size}")

    renderer.request_draw()


renderer.request_draw(lambda: renderer.render(scene, camera))

print(
    """Use the keys

 q  w  e
 a  s  d
 z  x  c

To change the anchor of the text.
For baseline anchoring, use Z X C (with Shift)

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
