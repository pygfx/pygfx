"""
Quickly visualize world objects with as
little boilerplate as possible.
"""

from .. import (
    Background,
    BackgroundMaterial,
    WgpuRenderer,
    WorldObject,
    Scene,
    PerspectiveCamera,
    OrbitControls,
)


def show(object: WorldObject):
    from wgpu.gui.auto import WgpuCanvas, run

    if isinstance(object, Scene):
        scene = object
    else:
        scene = Scene()
        scene.add(object)

        background = Background(None, BackgroundMaterial((0, 1, 0, 1), (0, 1, 1, 1)))
        scene.add(background)

    camera = PerspectiveCamera(70, 16 / 9)
    camera.show_object(object)

    canvas = WgpuCanvas()
    renderer = WgpuRenderer(canvas)

    controls = OrbitControls(camera.position.clone(), object.get_world_position())
    controls.add_default_event_handlers(canvas, camera)

    def animate():
        controls.update_camera(camera)
        renderer.render(scene, camera)

    canvas.request_draw(animate)
    run()
