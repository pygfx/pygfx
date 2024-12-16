"""
Nested Scenes
=============

Example of a scene rendered to a texture, which is shown in another scene.

Inception style. Since both the outer and inner scenes are lit, this
may look a bit weird. The faces of the outer cube are like "screens"
on which the subscene is displayed.

The sub-scene is rendered to a texture, and that texture is used for the
surface of the cube in the outer scene.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la


# First create the subscene, that reders into a texture

texture1 = gfx.Texture(dim=2, size=(200, 200, 1), format="rgba8unorm")

renderer1 = gfx.renderers.WgpuRenderer(texture1)
scene1 = gfx.Scene()

background1 = gfx.Background.from_color((0, 0.5, 0, 1))
scene1.add(background1)

im = iio.imread("imageio:bricks.jpg").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2)
geometry1 = gfx.box_geometry(200, 200, 200)
material1 = gfx.MeshPhongMaterial(map=tex, color=(1, 1, 0, 1.0))
cube1 = gfx.Mesh(geometry1, material1)
scene1.add(cube1)

camera1 = gfx.PerspectiveCamera(70, 16 / 9)
camera1.show_object(scene1)

scene1.add(gfx.AmbientLight(), camera1.add(gfx.DirectionalLight()))


# Then create the actual scene, in the visible canvas

canvas = RenderCanvas()  # for gallery scraper, bc we have 2 renderers
renderer2 = gfx.renderers.WgpuRenderer(canvas)
scene2 = gfx.Scene()

geometry2 = gfx.box_geometry(200, 200, 200)
material2 = gfx.MeshPhongMaterial(map=texture1)
cube2 = gfx.Mesh(geometry2, material2)
scene2.add(cube2)

camera2 = gfx.PerspectiveCamera(70, 16 / 9)
camera2.local.z = 400

scene2.add(gfx.AmbientLight(), camera2.add(gfx.DirectionalLight()))


def animate():
    rot = la.quat_from_euler((0.005, 0.01), order="xy")
    cube1.local.rotation = la.quat_mul(rot, cube1.local.rotation)
    cube2.local.rotation = la.quat_mul(rot, cube2.local.rotation)

    renderer1.render(scene1, camera1)
    renderer2.render(scene2, camera2)

    renderer2.request_draw()


if __name__ == "__main__":
    renderer2.request_draw(animate)
    loop.run()
