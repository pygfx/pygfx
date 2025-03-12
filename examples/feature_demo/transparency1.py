"""
Transparency 1
==============

Example showing transparency using three overlapping planes.
Press space to toggle the order of the planes.
Press 1-7 to select the blend mode.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

background = gfx.Background.from_color("#000")

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.4)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.4)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.4)))

plane1.local.position = (-10, -10, 1)
plane2.local.position = (0, 0, 2)
plane3.local.position = (10, 10, 3)

scene.add(background, plane1, plane2, plane3)

camera = gfx.OrthographicCamera(100, 100)

scene_overlay = gfx.Scene()
blend_mode_text = gfx.Text(
    text=f"Blend mode: {renderer.blend_mode}",
    anchor="bottom-left",
    material=gfx.TextMaterial(outline_thickness=0.3),
)
scene_overlay.add(blend_mode_text)

screen_camera = gfx.ScreenCoordsCamera()


@renderer.add_event_handler("key_down")
def handle_event(event):
    if event.key == " ":
        print("Rotating scene element order")
        scene.add(scene.children[1])  # skip bg
        canvas.request_draw()
    elif event.key == ".":
        clr = "#fff" if background.material.color_bottom_left == "#000" else "#000"
        print(f"Changing background color to {clr}")
        background.material.set_colors(clr)
        canvas.request_draw()
    elif event.key in "012345678":
        m = [
            None,  # 0
            "opaque",  # 1
            "dither",  # 2
            "ordered1",  # 3
            "ordered2",  # 4
            "weighted",  # 5
            "weighted_depth",  # 6
            "weighted_plus",  # 7
            "additive",  # 8
        ]
        mode = m[int(event.key)]
        renderer.blend_mode = mode
        print("Selecting blend_mode", mode)
        blend_mode_text.set_text(f"Blend mode: {mode}")


def animate():
    renderer.render(scene, camera, flush=False)
    renderer.render(scene_overlay, screen_camera, flush=True)


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    run()
