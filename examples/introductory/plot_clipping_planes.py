"""
Clipping Planes
===============

Example demonstrating clipping planes on a mesh.
"""
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


# Create a canvas and a renderer

canvas = WgpuCanvas(size=(800, 400))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Compose two of the same scenes


def create_scene(clipping_planes, clipping_mode):

    maxsize = 221
    scene = gfx.Scene()
    for n in range(20, maxsize, 50):
        material = gfx.MeshPhongMaterial(
            color=(n / maxsize, 1, 0, 1),
            clipping_planes=clipping_planes,
            clipping_mode=clipping_mode,
        )
        geometry = gfx.box_geometry(n, n, n)
        cube = gfx.Mesh(geometry, material)
        scene.add(cube)

    return scene


clipping_planes = [(-1, 0, 0, 0), (0, 0, -1, 0)]
scene1 = create_scene(clipping_planes, "any")
scene2 = create_scene(clipping_planes, "all")

scene1.add(gfx.AmbientLight(), gfx.DirectionalLight(position=(1, 2, 3)))
scene2.add(gfx.AmbientLight(), gfx.DirectionalLight(position=(1, 2, 3)))

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 250

controller = gfx.OrbitController(camera.position.clone())
controller.add_default_event_handlers(renderer, camera)


def animate():

    controller.update_camera(camera)

    w, h = canvas.get_logical_size()
    renderer.render(scene1, camera, flush=False, rect=(0, 0, w / 2, h))
    renderer.render(scene2, camera, flush=False, rect=(w / 2, 0, w / 2, h))
    renderer.flush()

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
