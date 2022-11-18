"""
Examples demonstrating text on dark and light background.

On the left plain text is shown in black and white. There is a
psychological effect that makes the bottom text (dark text on light
background) appear thinner than the other way around. The weight_offset
compensates for this effect.

On the right the text uses an outline to give a good appearance on any
background.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

scene = gfx.Scene()


scene.add(gfx.Background(None, gfx.BackgroundMaterial("#fff", "#000")))


t1 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum", font_size=40),
    gfx.TextMaterial(color="#fff", screen_space=True),
)
t2 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum", font_size=40),
    gfx.TextMaterial(color="#000", screen_space=True, weight_offset=50),
)
t3 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum", font_size=40),
    gfx.TextMaterial(
        color="#fff", screen_space=True, outline_color="#000", outline_thickness=0.15
    ),
)
t4 = gfx.Text(
    gfx.TextGeometry(text="Lorem ipsum", font_size=40),
    gfx.TextMaterial(
        color="#fff", screen_space=True, outline_color="#000", outline_thickness=0.15
    ),
)

t1.position.set(-1, +1, 0)
t2.position.set(-1, -1, 0)

t3.position.set(+1, +1, 0)
t4.position.set(+1, -1, 0)

scene.add(t1, t2, t3, t4)

camera = gfx.OrthographicCamera(4, 3)
camera.position.x = 0.8


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    run()
