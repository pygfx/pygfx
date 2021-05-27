"""
Example that implements a custom object and renders it.
"""

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas
from pyshader import python2shader
from pyshader import vec3

import pygfx as gfx


# %% Custom object, material, and matching render function


# This class has mostly a semantic purpose here
class Triangle(gfx.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


# Create a triangle material. We could e.g. define it's color here.
class TriangleMaterial(gfx.Material):
    pass


@python2shader
def vertex_shader(
    index: ("input", "VertexId", "i32"),
    pos: ("output", "Position", "vec4"),
    color: ("output", 0, "vec3"),
    stdinfo: ("uniform", (0, 0), gfx.renderers.wgpu.stdinfo_uniform_type),
):
    positions1 = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
    positions2 = [vec2(10.0, 10.0), vec2(90.0, 10.0), vec2(10.0, 90.0)]

    # Draw in NDC or screen coordinates
    # p = positions1[index]
    p = 2.0 * positions2[index] / stdinfo.logical_size - 1.0

    pos = vec4(p, 0.0, 1.0)  # noqa
    color = vec3(positions1[index], 0.5)  # noqa


@python2shader
def fragment_shader(
    in_color: ("input", 0, "vec3"),
    out_color: ("output", 0, "vec4"),
):
    out_color = vec4(in_color, 0.1)  # noqa


vertex_shader = """

[[block]]
struct Stdinfo {
    cam_transform: mat4x4<f32>;
    cam_transform_inv: mat4x4<f32>;
    projection_transform: mat4x4<f32>;
    projection_transform_inv: mat4x4<f32>;
    physical_size: vec2<f32>;
    logical_size: vec2<f32>;
};

[[group(0), binding(0)]]
var u_stdinfo: Stdinfo;

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] index: u32) -> [[builtin(position)]] vec4<f32> {
    let positions1 = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    let positions2 = array<vec2<f32>, 3>(vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0));

    // let p = positions1[index];
    let p = 2.0 * positions2[index] / u_stdinfo.logical_size - 1.0;
    return vec4<f32>(p, 0.0, 1.0);
}
"""

fragment_shader = """
struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] pick: vec4<i32>;
};

[[stage(fragment)]]
fn main() -> FragmentOutput {
    var out: FragmentOutput;
    out.color = vec4<f32>(1.0, 0.7, 0.2, 1.0);
    return out;
}
"""


# Tell pygfx to use this render function for a Triangle with TriangleMaterial.
@gfx.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(wobject, render_info):
    n = 3
    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": "triangle-list",
            "indices": range(n),
            "bindings0": {0: ("buffer/uniform", render_info.stdinfo_uniform)},
        },
    ]


# %% Setup scene

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
t1 = Triangle(TriangleMaterial())
scene.add(t1)
for i in range(2):
    scene.add(Triangle(TriangleMaterial()))

camera = gfx.NDCCamera()  # This material does not actually use the camera


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
