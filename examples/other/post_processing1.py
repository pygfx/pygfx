"""
Full-Screen Post Processing 1
=============================


Example full-screen post processing.

The idea is to render a scene to a texture, and then rendering
that texture as a full quad to the screen, while adding noise.

In many ways this example is similar to the scene_in_a_scene.py example,
except we use a custom object here for the noise.

Note that we may get a more streamlined way to implement post-processing effects.

"""

# sphinx_gallery_pygfx_docs = 'hidden'
# sphinx_gallery_pygfx_test = 'off'

import time

import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.renderers.wgpu import Binding, register_wgpu_render_function


raise RuntimeError("Post-processing needs to be redesigned")


# Create a custom object + material


class Fullquad(gfx.WorldObject):
    def __init__(self, texture, material):
        super().__init__()
        self.texture = texture
        self.material = material


class NoiseMaterial(gfx.materials.Material):
    # Note that this inherits fields from the base Material
    uniform_type = {
        "time": "f4",
        "noise": "f4",
    }

    def __init__(self, noise=1):
        super().__init__()

        self.uniform_buffer = gfx.Buffer(
            gfx.utils.array_from_shadertype(self.uniform_type)
        )
        self.uniform_buffer.data["time"] = 0
        self.uniform_buffer.data["noise"] = noise

    def tick(self):
        self.uniform_buffer.data["time"] = time.time() % 1
        self.uniform_buffer.update_full()


shader_source = """
struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) pick: vec4<i32>,
};

struct Render {
    opacity: f32,
    time: f32,
    noise: f32,
};
@group(0) @binding(0)
var<uniform> u_render: Render;

@group(1) @binding(0)
var r_sampler: sampler;
@group(1) @binding(1)
var r_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 4>(vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0));
    let pos = positions[index];
    var out: VertexOutput;
    out.texcoord = vec2<f32>(pos.x, 1.0 - pos.y);
    out.pos = vec4<f32>(pos * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let u_render_time = 0.0;
    let u_render_noise = 1.0;

    let xy = in.texcoord.xy;
    let random_nr = fract(sin(dot(xy, vec2<f32>(12.9898, 78.233)) + u_render.time) * 43758.5453);
    let noise = u_render.noise * random_nr;
    var out: FragmentOutput;
    out.color = textureSample(r_tex, r_sampler, xy) + vec4<f32>(noise, noise, noise, 1.0);
    return out;
}
"""


# Tell pygfx to use this render function for a Fullquad with NoiseMaterial.
@register_wgpu_render_function(Fullquad, NoiseMaterial)
def triangle_render_function(wobject, render_info):
    return [
        {
            "vertex_shader": (shader_source, "vs_main"),
            "fragment_shader": (shader_source, "fs_main"),
            "primitive_topology": "triangle-strip",
            "indices": 4,
            "bindings0": {
                0: Binding(
                    "u_render", "buffer/uniform", wobject.material.uniform_buffer
                ),
            },
            "bindings1": {
                0: Binding(
                    "r_sampler", "sampler/filtering", wobject.texture.get_view()
                ),
                1: Binding("r_tex", "texture/auto", wobject.texture.get_view()),
            },
        },
    ]


#  The application

# The canvas for eventual display
canvas = WgpuCanvas(size=(640, 480))

# The texture to render the scene into
texture = gfx.Texture(dim=2, size=(640, 480, 1), format="rgba8unorm")

# The regular scene

renderer1 = gfx.renderers.WgpuRenderer(texture)
scene = gfx.Scene()

im = iio.imread("imageio:astronaut.png").astype(np.float32) / 255
tex = gfx.Texture(im, dim=2).get_view(filter="linear", address_mode="repeat")

geometry = gfx.box_geometry(200, 200, 200)
material = gfx.MeshBasicMaterial(map=tex)
cube = gfx.Mesh(geometry, material)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

# The post processing scene

renderer2 = gfx.renderers.WgpuRenderer(canvas)
noise_object = Fullquad(texture, NoiseMaterial(0.2))


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    noise_object.material.tick()

    renderer1.render(scene, camera)
    renderer2.render(noise_object, gfx.NDCCamera())

    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
