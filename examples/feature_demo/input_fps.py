"""
First person shooter controls with GLFW GUI
=========================================

Example showing how to use the Input class to implement FPS controls.
"""

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import glfw
import numpy as np
import pylinalg as la
import pygfx as gfx
from wgpu.gui.glfw import WgpuCanvas, run


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Group()

# hide the cursor, lock it to the window and keep it centered
glfw.set_input_mode(canvas._window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# add background
background = gfx.Background(None, gfx.BackgroundMaterial("#ccc"))
scene.add(background)

# add floor
floor = gfx.Mesh(
    gfx.plane_geometry(20, 20),
    gfx.MeshPhongMaterial(color="#808080"),
)
floor.local.euler_x = np.pi * -0.5
floor.receive_shadow = True
scene.add(floor)

# add torus
torus = gfx.Mesh(
    gfx.torus_knot_geometry(1, 0.3, 128, 16),
    gfx.MeshPhongMaterial(color="#fff", side="Front"),
)
torus.local.y = 2
torus.cast_shadow = True
scene.add(torus)

# add lights
ambient = gfx.AmbientLight("#fff", 0.1)
scene.add(ambient)

light = gfx.DirectionalLight("#aaa")
light.local.x = -50
light.local.y = 50
light.cast_shadow = True
light.shadow.camera.width = 100
light.shadow.camera.height = 100
light.shadow.camera.update_projection_matrix()
scene.add(light)

# add camera
camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.local.position = 6, 2, 6
camera.look_at((0, 2, 0))

# add input
input = gfx.Input(register_events=renderer)
time = gfx.Time(register_events=renderer)

# fps controls state
velocity = 0.1
fly = False
mouse_sensitivity = 0.2
invert_y = -1
yaw, pitch = camera.local.euler_y, camera.local.euler_x


def clamp(x, a, b):
    return min(max(x, a), b)


def update():
    # FPS controls
    global yaw, pitch, fly

    # adjust settings
    if input.key_down("f"):
        fly = not fly

    # adjust camera angle based on mouse delta
    dx, dy = input.pointer_delta()
    if dx:
        yaw = (yaw + dx * mouse_sensitivity * time.delta * -1) % (np.pi * 2)
    if dy:
        pitch = clamp(
            pitch + dy * mouse_sensitivity * time.delta * invert_y,
            -np.pi * 0.49,
            np.pi * 0.49,
        )
    if dx or dy:
        camera.local.rotation = la.quat_from_euler([yaw, pitch], order="YX")

    # adjust camera position based on key state
    movement = np.array([0.0, 0.0, 0.0])
    if input.key("w"):
        movement += camera.local.forward * velocity
    if input.key("s"):
        movement += camera.local.forward * -velocity
    if input.key("a"):
        movement += camera.local.right * -velocity
    if input.key("d"):
        movement += camera.local.right * velocity
    if not fly:
        # project movement vector onto the floor
        movement -= (
            np.dot(movement, camera.local.reference_up) * camera.local.reference_up
        )
    camera.local.position = camera.local.position + movement


# setup animation loop
def animate():
    update()
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
