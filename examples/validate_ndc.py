"""
Example (and test) for the NDC coordinates. Draws a square that falls partly out of visible range.

* The scene should show a band from the bottom left to the upper right.
* The bottom-left (NDC -1 -1) must be green, the upper-right (NDC 1 1) blue.
* The other corners must be black, cut off at exactly half way: the depth is 0-1.

"""

from PySide6 import QtWidgets
from wgpu.gui.qt import WgpuCanvas

import pygfx as gfx


class Square(gfx.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


class SquareMaterial(gfx.Material):
    pass


shader_source = """
struct VertexOutput {
    [[location(0)]] color: vec4<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] index: u32) -> VertexOutput {
    var positions = array<vec3<f32>, 4>(
        vec3<f32>(-1.0, -1.0, 0.5), vec3<f32>(-1.0, 1.0, 1.5), vec3<f32>(1.0, -1.0, -0.5), vec3<f32>(1.0, 1.0, 0.5)
    );
    var colors = array<vec3<f32>, 4>(
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.5, 0.5), vec3<f32>(0.0, 0.5, 0.5), vec3<f32>(0.0, 0.0, 1.0)
    );

    var out: VertexOutput;
    out.pos = vec4<f32>(positions[index], 1.0);
    out.color = vec4<f32>(colors[index], 1.0);
    return out;
}

struct FragmentOutput {
    [[location(0)]] color: vec4<f32>;
    [[location(1)]] pick: vec4<i32>;
};

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.color = in.color;
    return out;
}
"""


@gfx.renderers.wgpu.register_wgpu_render_function(Square, SquareMaterial)
def square_render_function(wobject, render_info):
    return [
        {
            "vertex_shader": (shader_source, "vs_main"),
            "fragment_shader": (shader_source, "fs_main"),
            "primitive_topology": "triangle-strip",
            "indices": range(4),
        },
    ]


# %% Setup scene

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
t1 = Square(SquareMaterial())
scene.add(t1)

camera = gfx.NDCCamera()  # This example does not even use the camera


if __name__ == "__main__":
    print(__doc__)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    app.exec()
    canvas.closeEvent = lambda *args: app.quit()
