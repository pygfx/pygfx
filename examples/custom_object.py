"""
Example that implements a custom object and renders it.
"""

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

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


shader_source = """

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
var<uniform> u_stdinfo: Stdinfo;

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] index: u32) -> [[builtin(position)]] vec4<f32> {
    var positions1 = array<vec2<f32>, 3>(vec2<f32>(0.0, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.7));
    var positions2 = array<vec2<f32>, 3>(vec2<f32>(10.0, 10.0), vec2<f32>(90.0, 10.0), vec2<f32>(10.0, 90.0));

    // let p = positions1[index];
    let p = 2.0 * positions2[index] / u_stdinfo.logical_size - 1.0;
    return vec4<f32>(p, 0.0, 1.0);
}

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] pick: vec4<i32>;
};

[[stage(fragment)]]
fn fs_main() -> FragmentOutput {
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
            "vertex_shader": (shader_source, "vs_main"),
            "fragment_shader": (shader_source, "fs_main"),
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
for _ in range(2):
    scene.add(Triangle(TriangleMaterial()))

camera = gfx.NDCCamera()  # This material does not actually use the camera


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
    canvas.closeEvent = lambda *args: app.quit()
