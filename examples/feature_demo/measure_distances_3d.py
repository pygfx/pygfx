"""
Measure distances in 3D
=======================

Example to do measurements in a 3D scene. Click on the surface of the
two objects to see the distance betweeen these points.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2)

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
material = gfx.MeshPhongMaterial(map=tex, pick_write=True)

obj1 = gfx.Mesh(geometry, material)
obj2 = gfx.Mesh(geometry, material)
obj1.local.x = -3
obj2.local.x = 3
scene.add(obj1, obj2)

ruler = gfx.Ruler(ticks_at_end_points=True)
scene.add(ruler)

camera = gfx.PerspectiveCamera(70, 1)
camera.show_object(scene)

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

controller = gfx.OrbitController(camera, register_events=renderer)


@obj1.add_event_handler("pointer_down")
@obj2.add_event_handler("pointer_down")
def handle_clicks(event):
    if event.target is obj1 or event.target is obj2:
        face_index = event.pick_info["face_index"]
        face_coord = event.pick_info["face_coord"]
        vertex_indices = geometry.indices.data[face_index]
        pos = np.sum(
            [
                geometry.positions.data[i] * w
                for i, w in zip(vertex_indices, face_coord)
            ],
            axis=0,
        )
        pos += event.target.world.position

        if event.target is obj1:
            ruler.start_pos = pos
        else:
            ruler.end_pos = pos


def animate():
    ruler.update(camera, canvas.get_logical_size())
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
