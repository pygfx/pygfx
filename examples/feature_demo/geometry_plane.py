"""
Geometry Plane
==============

Use a plane geometry to show a texture, which is continuously updated to show video.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


def loop_video(video):
    while True:
        for frame in iio.imiter(video):
            yield frame[:, :, 0]


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

frame_generator = loop_video("imageio:cockatoo.mp4")
tex = gfx.Texture(next(frame_generator), dim=2)

geometry = gfx.plane_geometry(200, 200, 12, 12)
material = gfx.MeshBasicMaterial(map=tex)
plane = gfx.Mesh(geometry, material)
scene.add(plane)

camera = gfx.PerspectiveCamera(70)
camera.local.z = 200

scene.add(gfx.AmbientLight(), gfx.DirectionalLight())


def animate():
    # Read next frame, rewind if we reach the end
    tex.data[:] = next(frame_generator)
    tex.update_full()

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
