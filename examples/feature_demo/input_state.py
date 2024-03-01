"""
Input state
===========

Example showing how to use the Input class.
"""

# sphinx_gallery_pygfx_render = True
# sphinx_gallery_pygfx_target_name = "disp"

import numpy as np
import pylinalg as la
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Group()

# add background
background = gfx.Background(None, gfx.BackgroundMaterial("#ccc"))
scene.add(background)

# add floor
floor = gfx.Mesh(
    gfx.plane_geometry(20, 20),
    gfx.MeshPhongMaterial(color="#808080", side="Front"),
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
velocity = 0.1
mouse_sensitivity = 0.01
invert_x = -1
invert_y = -1


def update():
    # TODO: technically we also need the time delta since the last frame
    #       to be robust against varying frame rates
    # TODO: clamp pitch to (-90, 90) degrees
    # TODO: clamp pitch to (-90, 90) degrees
    # TODO: how to always capture mouse events even if the pointer is not over the window?
    # TODO: how to keep the mouse anchored to the center of the window?

    # adjust camera angle based on mouse delta
    rotation = input.pointer_delta()
    if rotation != (0, 0):
        pitch = la.quat_from_euler(
            rotation[1] * mouse_sensitivity * invert_y, order="X"
        )
        yaw = la.quat_from_euler(rotation[0] * mouse_sensitivity * invert_x, order="Y")
        camera.local.rotation = la.quat_mul(
            la.quat_mul(
                yaw,
                camera.local.rotation,
            ),
            pitch,
        )

    # implement basic FPS controls
    movement = np.array([0.0, 0.0, 0.0])
    if input.key("w"):
        movement += camera.local.forward * velocity
    if input.key("a"):
        movement += camera.local.right * -velocity
    if input.key("d"):
        movement += camera.local.right * velocity
    if input.key("s"):
        movement += camera.local.forward * -velocity

    # TODO: project movement vector onto floor (fly vs walk; toggle?)

    camera.local.position = camera.local.position + movement


# setup animation loop
def animate():
    update()
    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
