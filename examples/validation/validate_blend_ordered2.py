"""
Validate transparency for the ordered2 blend mode
=================================================

Display 4 semi-transparent planes::

            //             |         |         |         |
    camera ((              |         |         |         |
            \\             |         |         |         |

                         green1    red1     green2     red2

The order of rendering is: green1, red1, red2, green2

Since the latter two planes are out of order, they will be blended differently,
depending on the blend-mode.

There are also two vertical dark-blue strips. One opaque and one transparent.
These represent a dark object with an anti-aliased edge. This object is drawn
last, to demonstrate the dark artifacts from aa edges.

"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

# Note: we can either copy this example for different blend modes,
# or make this a multi-viewport example once we can use multiple blend-modes
# in one canvas.

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(600, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)
renderer.blend_mode = "ordered2"

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

geometry = gfx.plane_geometry(100, 50)
geometry_strip = gfx.plane_geometry(20, 200)

strip_opaque = gfx.Mesh(geometry_strip, gfx.MeshBasicMaterial(color=(0, 0, 0.5, 1)))
strip_transp = gfx.Mesh(geometry_strip, gfx.MeshBasicMaterial(color=(0, 0, 0.5, 0.5)))

plane_g1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.3)))
plane_g2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(0, 1, 0, 0.3)))
plane_r1 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))
plane_r2 = gfx.Mesh(geometry, gfx.MeshBasicMaterial(color=(1, 0, 0, 0.3)))

strip_opaque.local.position = 10, 0, 0
strip_transp.local.position = 21, 0, 0
strip_transp.local.scale_x = 0.1

plane_g1.local.position = 0, -25, 80
plane_r1.local.position = 0, 0, 60
plane_g2.local.position = 0, 25, 40
plane_r2.local.position = 0, 50, 20

scene.add(plane_g1, plane_r1, plane_r2, plane_g2)
scene.add(strip_opaque, strip_transp)

camera = gfx.OrthographicCamera(150, 150)
camera.local.position = (50, 5, 200)
camera.look_at((0, 10, 0))


canvas.request_draw(lambda: renderer.render(scene, camera))


if __name__ == "__main__":
    print(__doc__)
    run()
