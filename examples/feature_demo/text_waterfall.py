"""
Text waterfall
==============

An example showing a waterfall of text. On the left it shows the
contents of the glyph atlas. One goal of this example is to strain the
text rendering to its limits.
"""

# sphinx_gallery_pygfx_docs = 'animate 5s'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 400)))
scene = gfx.Scene()
glyph_atlas = gfx.utils.text.glyph_atlas
glyph_atlas.clear_free_regions = True  # So we can see regions being freed

# Add background
background = gfx.Background.from_color("#dde", "#fff")
scene.add(background)

# Add an image that shows the glyph atlas
atlas_viewer = gfx.Mesh(
    gfx.plane_geometry(100, 100),
    gfx.MeshBasicMaterial(color="red"),
)
scene.add(atlas_viewer)
atlas_viewer.local.x = -50

camera = gfx.OrthographicCamera(200, 100)

# Create a bunch of reusable text objects


def character_generator():
    pieces = gfx.font_manager.select_font(" ", gfx.font_manager.default_font_props)
    font = pieces[0][1]
    while True:
        for c in font.codepoints:
            yield chr(c)


chargen = character_generator()

live_objects = set()
waiting_objects = set()

text_material = gfx.TextMaterial(color="#06E")

for _i in range(100):
    obj = gfx.Text(text=" ", font_size=18, screen_space=True, material=text_material)
    scene.add(obj)
    waiting_objects.add(obj)
    obj.local.y = -999


# The animate function makes the text objects fall down, and update the objects
# with a new character once they start their fall again.
# Until we have real garbage collection for glyphs, we fake it here.


def animate():
    garbage_collect = True

    # Let them fall
    for obj in list(live_objects):
        obj.local.y -= obj.fall_speed
        if obj.local.y < -60:
            live_objects.discard(obj)
            waiting_objects.add(obj)
            if garbage_collect:
                all_indices = set()
                for x in live_objects:
                    atlas_indices = x.geometry.glyph_data.data["atlas_index"]
                    all_indices.update(int(index) for index in atlas_indices)
                for index in obj.geometry.glyph_data.data["atlas_index"]:
                    index = int(index)
                    if index not in all_indices:
                        glyph_atlas.free_region(index)

    # Drop new objects
    if waiting_objects:
        obj = waiting_objects.pop()
        live_objects.add(obj)
        obj.local.y = 50
        obj.local.x = np.random.uniform(0, 100)
        obj.set_text(next(chargen))
        obj.fall_speed = np.random.uniform(1, 4)

    # Update the image
    if atlas_viewer.material.map is not glyph_atlas.texture:
        atlas_viewer.material.map = glyph_atlas.texture

    # Render
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
