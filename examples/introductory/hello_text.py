"""
Hello text
==========

Example showing text in world and screen space. Also notice how the
text is shown on top of the plane (due to its depth offset).
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

from time import perf_counter

import pygfx as gfx
import pylinalg as la


scene = gfx.Scene()

# Create a plane to put attach text to
plane = gfx.Mesh(
    gfx.box_geometry(20, 20, 1),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)
scene.add(plane)

# Create two texts, one on each side of the plane
text1 = gfx.Text(
    text="Hello world",
    font_size=2.8,
    material=gfx.TextMaterial(color="#ddd"),
)
text1.local.position = (0, 0, 0.5)
plane.add(text1)

text2 = gfx.Text(
    text="Здравей свят",
    font_size=2.0,
    material=gfx.TextMaterial(color="#ddd"),
)
text2.local.position = (0, 0, -0.5)
text2.local.scale = (-1, 1, 1)
plane.add(text2)

# Another text in screen space. Also shows markdown formatting
text3 = gfx.Text(
    markdown=" **Screen** space",
    screen_space=True,
    font_size=20,
    anchor="bottom-left",
    material=gfx.TextMaterial(color="#0f4"),
)
text3.local.position = (10, 10, 0)
plane.add(text3)

# Let there be ...
scene.add(gfx.AmbientLight())
light = gfx.DirectionalLight()
light.local.position = (0, 0, 1)
scene.add(light)


t_last = perf_counter()


def before_render():
    global t_last
    t_new = perf_counter()
    dt = t_new - t_last
    t_last = t_new
    rot = la.quat_from_euler((0.25 * dt, 0.5 * dt), order="XY")
    plane.local.rotation = la.quat_mul(rot, plane.local.rotation)


disp = gfx.Display(before_render=before_render)

if __name__ == "__main__":
    disp.show(scene)
