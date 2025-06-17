"""
Post processing effects 1
=========================

Example post-processing effects, showing builtin effects.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
from pygfx.renderers.wgpu import NoisePass, FogPass, FXAAPass

canvas = RenderCanvas(update_mode="continuous")
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.effect_passes = [NoisePass(0.5), FogPass("#fff"), FXAAPass()]

scene = gfx.Scene()

im = iio.imread("imageio:bricks.jpg")
tex = gfx.Texture(im, dim=2)

geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
geometry.texcoords.data[:, 0] *= 10  # stretch the texture
obj = gfx.Mesh(
    geometry,
    gfx.MeshPhongMaterial(map=tex),
)
scene.add(obj)

camera = gfx.OrthographicCamera(depth_range=(-2, 2))
camera.show_object(scene)

scene.add(gfx.AmbientLight(), camera.add(gfx.DirectionalLight()))

controller = gfx.TrackballController(camera, register_events=renderer)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    loop.run()
