"""
Text markdown formatting
========================

Test markdown formatting features.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run

import pygfx as gfx

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#aaa"))

md = """
## H2 header
### H3 header

A bullet list:
  * Bullet 1
  - Bullet 2
**bold** normal, loose ** stays
normal *italic*, loose * stays
**whole **sentense** bold.** not this
also partial**bold**wordpart
"""

material = gfx.TextMaterial(color="#000")

text1 = gfx.Text(anchor="top-left", material=material)
text2 = gfx.Text(anchor="top-left", material=material)
text3 = gfx.Text(anchor="top-left", material=material)
text4 = gfx.Text(anchor="top-left", material=material)

# Set the markdown as markdown or as text. Also use the variation where
# it results in a single TextBlock, which should result in the same result.
text1.set_markdown("# md multi-block\n" + md)
text2.set_markdown(["# md single-block", md])
text3.set_text("--- text multi-block:\n" + md)
text4.set_text(["--- text single-block:", md])

text1.local.position = (0, 0, 0)
text2.local.position = (250, 0, 0)
text3.local.position = (0, -200, 0)
text4.local.position = (250, -200, 0)

scene.add(text1, text2, text3, text4)

camera = gfx.OrthographicCamera(1, 1)

camera.show_object(scene, scale=0.7)
renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))

renderer.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
