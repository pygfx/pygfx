"""
Transparency 2
==============

Example showing transparency using three orthogonal planes.
Press space to toggle the order of the planes.
Press 1-7 to select the blend mode.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas, pixel_ratio=2)
scene = gfx.Scene()

background = gfx.Background.from_color("#000")

sphere = gfx.Mesh(gfx.sphere_geometry(10), gfx.MeshPhongMaterial())

geometry = gfx.plane_geometry(50, 50)
plane1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))
plane2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.5)))
plane3 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 0, 1, 0.7)))

plane1.local.rotation = la.quat_from_axis_angle((1, 0, 0), 1.571)
plane2.local.rotation = la.quat_from_axis_angle((0, 1, 0), 1.571)
plane3.local.rotation = la.quat_from_axis_angle((0, 0, 1), 1.571)

scene.add(background, plane1, plane2, plane3, sphere)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.show_object(scene, view_dir=(-1, -2, -3))
controller = gfx.OrbitController(camera, register_events=renderer)

scene.add(camera.add(gfx.DirectionalLight()))

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
    renderer.render(
        scene_overlay,
        screen_camera,
        flush=True,
    )


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(animate)
    run()
