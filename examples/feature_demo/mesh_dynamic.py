"""
Mesh dynamic
============

Example showing a Torus knot, dynamically changing what faces are shown.
"""
# sphinx_gallery_pygfx_render = True

import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2)

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 16)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshPhongMaterial(map=tex)
obj = gfx.Mesh(geometry, material)
scene.add(obj)

camera = gfx.PerspectiveCamera(70, 1)
camera.show_object(scene)

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

controller = gfx.TrackballController(camera, register_events=renderer)


def animate():
    indices = obj.geometry.indices

    # Update the view params
    offset = indices.view[0] + 32
    if offset + 640 >= indices.nitems:
        offset = 0
    indices.view = offset, 640

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
