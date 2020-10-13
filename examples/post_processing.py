"""
Example full-screen post processing.
"""

import time

import numpy as np
import imageio
import pyshader
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


app = QtWidgets.QApplication([])


class NoisyPostProcessingStep(gfx.renderers.wgpu.PostProcessingStep):
    def __init__(self, device):
        super().__init__(
            device,
            fragment_shader=noise_fragment_shader,
            uniform_type=noise_uniform_type,
        )

    def render(self, src, dst):
        self._uniform_data["time"] = time.perf_counter()
        self._uniform_data["noise"] = 0.03
        super().render(src, dst)


noise_uniform_type = pyshader.Struct(time=pyshader.f32, noise=pyshader.f32)


@pyshader.python2shader
def noise_fragment_shader(
    v_texcoord: (pyshader.RES_INPUT, 0, "vec2"),
    s_sam: (pyshader.RES_SAMPLER, (0, 0), ""),
    t_tex: (pyshader.RES_TEXTURE, (0, 1), "2d f32"),
    u_render: (pyshader.RES_UNIFORM, (0, 2), noise_uniform_type),
    out_color: (pyshader.RES_OUTPUT, 0, "vec4"),
):
    val = t_tex.sample(s_sam, v_texcoord)

    # todo: use spirv builtin random number gen
    xy = v_texcoord.xy
    noise = fract(sin(xy @ vec2(12.9898, 78.233) + u_render.time) * 43758.5453)
    noise = u_render.noise * noise
    val += vec4(noise, noise, noise, 1.0)
    out_color = val  # noqa - shader output


canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

renderer.post_processing_steps.append(NoisyPostProcessingStep(renderer.device))

im = imageio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2, usage="sampled").get_view(
    filter="linear", address_mode="repeat"
)

geometry = gfx.BoxGeometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)


camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    app.exec_()
