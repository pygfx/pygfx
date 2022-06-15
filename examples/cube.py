"""
Example showing a single geometric cube.
"""

from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


renderer = gfx.renderers.WgpuRenderer(WgpuCanvas())
scene = gfx.Scene()

cube = gfx.Mesh(
    gfx.box_geometry(200, 200, 200),
    gfx.MeshPhongMaterial(color="#336699"),
)
scene.add(cube)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400

m1 = cube.material
m2 = gfx.MeshPhongMaterial(color="#ff6699")
m3 = gfx.MeshBasicMaterial(color="#11ff99")

p1 = cube.geometry.positions
p2 = gfx.Buffer(cube.geometry.positions.data * 0.7)

g1 = cube.geometry
g2 = gfx.Geometry()
g2.positions = p2
g2.normals = g1.normals
g2.indices = g1.indices


def animate():
    rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
    cube.rotation.multiply(rot)

    renderer.render(scene, camera)
    renderer.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    run()
