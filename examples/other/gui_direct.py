"""
Direct integration of glfw and wgpu-py without using the RenderCanvas library.

Demonstration for hardcore users that need total low-level control.

"""

# run_example = false

import os
import sys
import time
import atexit

import glfw
import wgpu
import pygfx as gfx


system_is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower()
api_is_wayland = False
if sys.platform.startswith("linux") and system_is_wayland:
    if not hasattr(glfw, "get_x11_window"):
        api_is_wayland = True


def get_glfw_present_info(window):
    if sys.platform.startswith("win"):
        return {
            "platform": "windows",
            "window": int(glfw.get_win32_window(window)),
            "vsync": True,
        }
    elif sys.platform.startswith("darwin"):
        return {
            "platform": "cocoa",
            "window": int(glfw.get_cocoa_window(window)),
            "vsync": True,
        }
    elif sys.platform.startswith("linux"):
        if api_is_wayland:
            return {
                "platform": "wayland",
                "window": int(glfw.get_wayland_window(window)),
                "display": int(glfw.get_wayland_display()),
                "vsync": True,
            }
        else:
            return {
                "platform": "x11",
                "window": int(glfw.get_x11_window(window)),
                "display": int(glfw.get_x11_display()),
                "vsync": True,
            }
    else:
        raise RuntimeError(f"Cannot get GLFW surface info on {sys.platform}.")


# Setup glfw
glfw.init()
atexit.register(glfw.terminate)

# disable automatic API selection, we are not using opengl
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.RESIZABLE, True)


title = "wgpu glfw direct"
window = glfw.create_window(640, 480, title, None, None)
present_info = get_glfw_present_info(window)

context = wgpu.gpu.get_canvas_context(present_info)


def _on_size_change(*args):
    renderer.pixel_ratio = glfw.get_window_content_scale(window)[0]
    renderer.physical_size = glfw.get_framebuffer_size(window)


glfw.set_framebuffer_size_callback(window, _on_size_change)
glfw.set_window_content_scale_callback(window, _on_size_change)


renderer = gfx.WgpuRenderer(context)
camera = gfx.NDCCamera()

triangle = gfx.Mesh(
    gfx.Geometry(
        indices=[(0, 1, 2)],
        positions=[(0.0, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.75, 0)],
        colors=[(1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
    ),
    gfx.MeshBasicMaterial(color_mode="vertex"),
)

# Initialize physical size once. For robust apps update this on resize events.
_on_size_change()


def main():
    last_frame_time = time.perf_counter()
    frame_count = 0

    # render loop
    while not glfw.window_should_close(window):
        # process inputs
        glfw.poll_events()

        # draw a frame
        renderer.render(triangle, camera)

        # present the frame to the screen
        context.present()

        # stats
        frame_count += 1
        etime = time.perf_counter() - last_frame_time
        if etime > 1:
            print(f"{frame_count / etime:0.1f} FPS")
            last_frame_time, frame_count = time.perf_counter(), 0

    # dispose resources
    context.unconfigure()
    glfw.destroy_window(window)

    # allow proper cleanup (workaround for glfw bug)
    end_time = time.perf_counter() + 0.1
    while time.perf_counter() < end_time:
        glfw.wait_events_timeout(end_time - time.perf_counter())


if __name__ == "__main__":
    main()
