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
# H1 header
## H2 header
### H3 header

A bullet list:
  * Bullet 1
  - Bullet 2
**bold** normal, loose ** stays
normal *italic*, loose * stays
**whole **sentense** bold.** not this
also partial**bold** and **bold**partial
"""

material = gfx.TextMaterial(color="#000")

text1 = gfx.Text(gfx.TextGeometry(space_mode="model"), material)
text2 = gfx.Text(gfx.TextGeometry(space_mode="model"), material)
text3 = gfx.Text(gfx.TextGeometry(space_mode="model"), material)
text4 = gfx.Text(gfx.TextGeometry(space_mode="model"), material)

text1.local.position = (0, 0, 0)
text2.local.position = (250, 0, 0)
text3.local.position = (0, -190, 0)
text4.local.position = (250, -190, 0)

text1.geometry.set_markdown(md)
text2.geometry.set_text_block_count(1)
text2.geometry.get_text_block(0).set_markdown(md)
text3.geometry.set_text(md)
text4.geometry.set_text_block_count(1)
text4.geometry.get_text_block(0).set_text(md)

for ob in (text1, text2, text3, text4):
    ob.geometry.anchor = "top-left"
    ob.geometry._on_update_object()

scene.add(text1, text2, text3, text4)

camera = gfx.OrthographicCamera(1, 1)

camera.show_object(scene, scale=0.7)
renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))


renderer.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
