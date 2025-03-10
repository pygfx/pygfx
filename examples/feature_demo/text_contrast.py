"""
Text contrast
=============

Example demonstrating text on dark and light background.

On the left, plain text is shown in black and white. There is a
psychological effect that makes the bottom text appear thinner than the
other way around  (dark text on light background). The weight_offset
compensates for this effect.

On the right the text uses an outline to give a good appearance on any
background.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#fff", "#000"))


t1 = gfx.Text(
    text="Lorem ipsum",
    font_size=40,
    screen_space=True,
    material=gfx.TextMaterial(color="#fff"),
)
t2 = gfx.Text(
    text="Lorem ipsum",
    font_size=40,
    screen_space=True,
    material=gfx.TextMaterial(color="#000", weight_offset=50),
)
t3 = gfx.Text(
    text="Lorem ipsum",
    font_size=40,
    screen_space=True,
    material=gfx.TextMaterial(
        color="#fff", outline_color="#000", outline_thickness=0.15
    ),
)
t4 = gfx.Text(
    text="Lorem ipsum",
    font_size=40,
    screen_space=True,
    material=gfx.TextMaterial(
        color="#fff", outline_color="#000", outline_thickness=0.15
    ),
)

t1.local.position = (-1, +1, 0)
t2.local.position = (-1, -1, 0)

t3.local.position = (+1, +1, 0)
t4.local.position = (+1, -1, 0)

scene.add(t1, t2, t3, t4)

camera = gfx.OrthographicCamera(4, 3)


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    run()
