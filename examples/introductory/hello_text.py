"""
Hello text
==========

Example showing text in world and screen space.
"""

# sphinx_gallery_pygfx_animate = True
# sphinx_gallery_pygfx_target_name = "disp"


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
    gfx.TextGeometry("Hello world", font_size=2.8),
    gfx.TextMaterial(color="#ddd"),
)
text1.local.position = (0, 0, 0.55)
plane.add(text1)

text2 = gfx.Text(
    gfx.TextGeometry("Здравей свят", font_size=2.8),
    gfx.TextMaterial(color="#ddd"),
)
text2.local.position = (0, 0, -0.55)
text2.local.scale = (-1, 1, 1)
plane.add(text2)

# Another text in screen space. Also shows markdown formatting
text3 = gfx.Text(
    gfx.TextGeometry(
        markdown=" **Screen** space",
        screen_space=True,
        font_size=20,
        anchor="bottomleft",
    ),
    gfx.TextMaterial(color="#0f4"),
)
text3.local.position = (10, 10, 0)
plane.add(text3)

# Let there be ...
scene.add(gfx.AmbientLight())
light = gfx.DirectionalLight()
light.local.position = (0, 0, 1)
scene.add(light)


def before_render():
    rot = la.quat_from_euler((0.005, 0.01), order="XY")
    plane.local.rotation = la.quat_mul(rot, plane.local.rotation)


disp = gfx.Display(before_render=before_render)

if __name__ == "__main__":
    disp.show(scene)
