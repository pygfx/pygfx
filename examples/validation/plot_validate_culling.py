"""
Culling
=======

Example test to validate winding and culling.

* The top red knot should look normal and well lit.
* The top green know should show the backfaces, well lit.
* The bottom row shows the same, but the camera looks backwards.

"""
# test_example = true
# sphinx_gallery_pygfx_render = True

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

# geometry = gfx.BoxGeometry(1, 1, 1)
geometry = gfx.torus_knot_geometry(1, 0.3, 64, 10)

# Create red know shown normally
material1 = gfx.MeshPhongMaterial(color=(1, 0, 0, 1))
obj1 = gfx.Mesh(geometry, material1)
obj1.position.set(-2, 0, 0)
obj1.material.side = "FRONT"

# Create a green knot for which we show the back
material2 = gfx.MeshPhongMaterial(color=(0, 1, 0, 1))
obj2 = gfx.Mesh(geometry, material2)
obj2.position.set(+2, 0, 0)
obj2.material.side = "BACK"

# Rotate all of them and add to scene
rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.71, 1))
obj1.rotation.multiply(rot)
obj2.rotation.multiply(rot)
scene.add(obj1, obj2)

camera = gfx.OrthographicCamera(6, 4)

dir_light = gfx.DirectionalLight(1, 2)
dir_light.position.set(0, 0, 1)

scene.add(gfx.AmbientLight(1, 0.2), dir_light)


def animate():
    # Render top row
    camera.scale.z = 1
    renderer.render(scene, camera, rect=(0, 0, 600, 300), flush=False)
    # Render same scene in bottom row. The camera's z scale is negative.
    # This means it looks backwards, but more importantly, it means that the
    # winding is affected. The result should still look correct because we
    # take this effect into account in the mesh shader.
    camera.scale.z = -1
    renderer.render(scene, camera, rect=(0, 300, 600, 300))


canvas.request_draw(animate)


if __name__ == "__main__":
    print(__doc__)
    run()
