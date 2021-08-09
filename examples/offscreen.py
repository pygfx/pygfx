"""
Example demonstrating off-screen rendering.
"""

import imageio
import pygfx as gfx

# Instead of a canvas, we provide a size (in logical pixels). Properties
# like line widths are usually expressed relative to this logical size.
# We set pixel_ratio to 2 (the default) to double the resolution.
renderer = gfx.renderers.WgpuRenderer(size=(640, 480), pixel_ratio=2)
scene = gfx.Scene()

im = imageio.imread("imageio:astronaut.png").astype("float32") / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.5, 1.0))
cube.rotation.multiply(rot)


if __name__ == "__main__":
    renderer.render(scene, camera)
    im = renderer.snapshot()
    print(im.shape)  # (960, 1280, 4)
    imageio.imsave("offscreen.png", im)
