"""
Example that implements a custom object and renders it.
"""

from PyQt5 import QtWidgets
from wgpu.gui.qt import WgpuCanvas
from python_shader import python2shader, RES_INPUT, RES_OUTPUT, RES_UNIFORM
from python_shader import vec3

import visvis2 as vv


# %% Custom object, material, and matching render function


# This class has mostly a semantic purpose here
class Triangle(vv.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


# Create a triangle material. We could e.g. define it's color here.
class TriangleMaterial(vv.Material):
    pass


@python2shader
def vertex_shader(
    index: (RES_INPUT, "VertexId", "i32"),
    pos: (RES_OUTPUT, "Position", "vec4"),
    color: (RES_OUTPUT, 0, "vec3"),
    stdinfo: (RES_UNIFORM, (0, 0), vv.renderers.wgpu.stdinfo_uniform_type),
):
    # Draw in NDC
    # positions = [vec2(+0.0, -0.5), vec2(+0.5, +0.5), vec2(-0.5, +0.7)]
    # p = positions[index]

    # Draw in screen coordinates
    positions = [vec2(10.0, 10.0), vec2(90.0, 10.0), vec2(10.0, 90.0)]
    p = 2.0 * positions[index] / stdinfo.logical_size - 1.0

    pos = vec4(p, 0.0, 1.0)  # noqa
    color = vec3(p, 0.5)  # noqa


@python2shader
def fragment_shader(
    in_color: (RES_INPUT, 0, "vec3"), out_color: (RES_OUTPUT, 0, "vec4"),
):
    out_color = vec4(0.0, 1.0, 1.0, 0.1)  # noqa


# Tell visvis to use this render function for a Triangle with TriangleMaterial.
@vv.renderers.wgpu.register_wgpu_render_function(Triangle, TriangleMaterial)
def triangle_render_function(wobject):
    n = 3
    return [
        {
            "vertex_shader": vertex_shader,
            "fragment_shader": fragment_shader,
            "primitive_topology": "triangle-list",
            "indices": range(n),
        },
    ]


# %% Setup scene

app = QtWidgets.QApplication([])

canvas = WgpuCanvas()
renderer = vv.WgpuRenderer(canvas)

scene = vv.Scene()
t1 = Triangle(TriangleMaterial())
scene.add(t1)
for i in range(2):
    scene.add(Triangle(TriangleMaterial()))

camera = vv.NDCCamera()  # This material does not actually use the camera


def animate():
    width, height, ratio = canvas.get_size_and_pixel_ratio()
    camera.set_viewport_size(width, height)
    renderer.render(scene, camera)


if __name__ == "__main__":
    canvas.draw_frame = animate
    app.exec_()
    canvas.closeEvent = lambda *args: app.quit()
