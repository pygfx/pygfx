"""
Culling
=======

Example test to validate winding and culling.

* The rightmost shapes are only lit by a dim ambient light. Other than that:
* The top green shapes should look normal and well lit.
* The bottom yellow shapes should show the backfaces, well lit.
* Hit space to toggle flat shading.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import pylinalg as la


canvas = WgpuCanvas(size=(1500, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# geometry = gfx.BoxGeometry(1, 1, 1)
geometry = gfx.torus_knot_geometry(1, 0.3, 256, 20)


def create_scene(title):
    # Create a green knot shown normally
    material1 = gfx.MeshPhongMaterial(color=(0, 1, 0, 1))
    obj1 = gfx.Mesh(geometry, material1)
    obj1.local.position = (0, 2, 0)
    obj1.material.side = gfx.VisibleSide.front

    # Create a yellow knot for which we show the back
    material2 = gfx.MeshPhongMaterial(color=(1, 1, 0, 1))
    obj2 = gfx.Mesh(geometry, material2)
    obj2.local.position = (0, -2, 0)
    obj2.material.side = gfx.VisibleSide.back

    # Rotate both in a position where the back faces are easier spotted
    rot = la.quat_from_euler((0.71, 1), order="XY")
    obj1.local.rotation = la.quat_mul(rot, obj1.local.rotation)
    obj2.local.rotation = la.quat_mul(rot, obj2.local.rotation)

    t = gfx.Text(text=title, screen_space=True, font_size=20)
    t.local.position = (0, 4, 0)
    camera = gfx.OrthographicCamera(4.2, 6)

    amb_light = gfx.AmbientLight(0.2, 2)
    dir_light = gfx.DirectionalLight(1, 2)

    # Compose scene
    scene = gfx.Scene()
    scene.add(obj1, obj2, t, amb_light, camera.add(dir_light))

    return scene


vp1 = gfx.Viewport(renderer, (0, 0, 300, 600))
scene1 = create_scene("Regular")

vp2 = gfx.Viewport(renderer, (300, 0, 300, 600))
scene2 = create_scene("Flip object")
scene2.children[0].local.scale_x = -1
scene2.children[1].local.scale_x = -1

vp3 = gfx.Viewport(renderer, (600, 0, 300, 600))
scene3 = create_scene("Rotate camera")
transform = scene3.children[-1].local
transform.rotation = la.quat_from_axis_angle((0, 1, 0), 3.141592)

vp4 = gfx.Viewport(renderer, (900, 0, 300, 600))
scene4 = create_scene("Flip camera")
scene4.children[-1].local.scale_z = -1

vp5 = gfx.Viewport(renderer, (1200, 0, 300, 600))
scene5 = create_scene("Flip camera, not light")
scene5.children[-1].local.scale_z = -1
scene5.add(scene5.children[-1].children[0])  # put light in root
scene5.add(scene5.children[-2])  # move camera to the end


@canvas.add_event_handler("key_down")
def toggle_flat_shading(e):
    if e["key"] == " ":
        for scene in (scene1, scene2, scene3, scene4, scene5):
            for obj in scene.children[:2]:
                obj.material.flat_shading = not obj.material.flat_shading
        canvas.request_draw()


def animate():
    vp1.render(scene1, scene1.children[-1])
    vp2.render(scene2, scene2.children[-1])
    vp3.render(scene3, scene3.children[-1])
    vp4.render(scene4, scene4.children[-1])
    vp5.render(scene5, scene5.children[-1])
    renderer.flush()


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
