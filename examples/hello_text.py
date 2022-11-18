"""
Example showing text in world and screen space.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()


# Create a plane to put attach text to
plane = gfx.Mesh(
    gfx.box_geometry(20, 20, 1),
    gfx.MeshPhongMaterial(color=(0.2, 0.4, 0.6, 1.0)),
)
scene.add(plane)

# Create two texts, one on each side of the plane
text1 = gfx.Text(
    gfx.TextGeometry("Привет мир", font_size=3),
    gfx.TextMaterial(color="#ddd", screen_space=False),
)
text1.position.set(-9, 0, 0.55)
plane.add(text1)

text2 = gfx.Text(
    gfx.TextGeometry("Hello world", font_size=3),
    gfx.TextMaterial(color="#ddd", screen_space=False),
)
text2.position.set(9, 0, -0.55)
text2.scale.set(-1, 1, 1)
plane.add(text2)

# Another text in screen space. Also shows markdown formatting
text3 = gfx.Text(
    gfx.TextGeometry(markdown="**Screen** space", font_size=20),
    gfx.TextMaterial(color="#0f4", screen_space=True),
)
text3.position.set(10, 0, 0)
plane.add(text3)

# Let there be ...
scene.add(gfx.AmbientLight())
scene.add(gfx.DirectionalLight(position=(0, 0, 1)))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 25


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    plane.rotation.multiply(rot)
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
