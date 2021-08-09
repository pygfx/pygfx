"""
Example full-screen post processing.
"""

import time

import numpy as np
import imageio
import pygfx as gfx

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas


class NoisyPostProcessingStep(gfx.renderers.wgpu.PostProcessingStep):
    def __init__(self, device):
        super().__init__(
            device,
            fragment_shader=noise_fragment_shader,
            uniform_type=noise_uniform_type,
        )

    def render(self, src, dst):
        self._uniform_data["time"] = time.perf_counter()
        self._uniform_data["noise"] = 0.1
        super().render(src, dst)


noise_uniform_type = dict(time=("float32",), noise=("float32",))


noise_fragment_shader = """
struct VertexOutput {
    [[location(0)]] texcoord: vec2<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};
[[group(0), binding(0)]]
var r_sampler: sampler;
[[group(0), binding(1)]]
var r_tex: texture_2d<f32>;

[[block]]
struct Render {
    time: f32;
    noise: f32;
};
[[group(0), binding(2)]]
var u_render: Render;

[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let xy = in.texcoord.xy;
    let random_nr = fract(sin(dot(xy, vec2<f32>(12.9898, 78.233)) + u_render.time) * 43758.5453);
    let noise = u_render.noise * random_nr;
    return textureSample(r_tex, r_sampler, xy) + vec4<f32>(noise, noise, noise, 1.0);
}
"""


app = QtWidgets.QApplication([])
canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

renderer.postfx.append(NoisyPostProcessingStep(renderer.device))

im = imageio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

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
