"""
Example that implements a custom object and renders it.
"""

from PyQt5 import QtWidgets
import wgpu
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
    in_color: ("input", 0, "vec3"), out_color: ("output", 0, "vec4"),
):
    out_color = vec4(in_color, 0.1)  # noqa


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
            "bindings0": {0: (wgpu.BindingType.uniform_buffer, render_info.stdinfo)},
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


def animate():
    physical_size = canvas.get_physical_size()
    camera.set_viewport_size(*physical_size)
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
