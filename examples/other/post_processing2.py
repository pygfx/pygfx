"""
Fullscreen Postprocessing 2
===========================

Example showing post-processing effects by modifying the flusher object.

This example is a placeholder for how post-processing *could* work if
we'd provide an API for it.

Note: this example makes heavy use of private variables and makes
assumptions about how the RenderFlusher works that may not hold in the
future.

"""

# sphinx_gallery_pygfx_docs = 'hidden'
# sphinx_gallery_pygfx_test = 'off'

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


raise RuntimeError("Post-processing needs to be redesigned")

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:bricks.jpg").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.box_geometry(200, 200, 200)
geometry.texcoords.data[:] *= 2  # smaller bricks
material = gfx.MeshPhongMaterial(map=tex, color=(1, 0, 0, 0.2))
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


class MyRenderFlusher(gfx.renderers.wgpu._renderutils.RenderFlusher):
    uniform_type = dict(
        size="2xf4",
        sigma="f4",
        support="i4",
        amplitude="f4",
    )

    def __init__(self, device):
        super().__init__(device)
        code = """
            let a = u_render.amplitude;
            tex_coord.x = tex_coord.x + sin(tex_coord.y * 20.0) * a;
        """
        self._shader["tex_coord_map"] = code
        self._uniform_data["amplitude"] = 0.02


renderer._flusher = MyRenderFlusher(renderer.device)


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
