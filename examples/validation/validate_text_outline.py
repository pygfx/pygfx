"""
Text outline always behind text itself
======================================

Example demonstrating the capabilities of text outline to stay behind the text
itself, even for very thick value of the outline thickness where the outline of
one character may overlap with the neighboring one.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#fff", "#000"))

text = gfx.Text(
    text="Hello world",
    font_size=100,
    screen_space=True,
    text_align="right",
    anchor="middle-center",
    material=gfx.TextMaterial(
        color="#DA9DFF",
        outline_color="#000",
        # Choose a very thick outline to ensure the effect is noticeable
        outline_thickness=0.45,
    ),
)

scene.add(text)

camera = gfx.OrthographicCamera(4, 3)
renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))

renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    run()
