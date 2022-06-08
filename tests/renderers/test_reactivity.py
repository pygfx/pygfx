import numpy as np
import wgpu
import pygfx as gfx


from pygfx.renderers.wgpu._pipelinebuilder import ensure_pipeline


renderer = gfx.renderers.WgpuRenderer(
    gfx.Texture(dim=2, size=(10, 10, 1), format=wgpu.TextureFormat.rgba8unorm)
)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.position.z = 400


def peek_labels(wobject):
    result = set()
    for labels in wobject.tracker._trackable_changed.values():
        result.update(labels)
    return set(label.lstrip("!") for label in result)


def render(wobject):
    labels = peek_labels(wobject)
    renderer.render(wobject, camera)
    assert peek_labels(wobject) == set()
    return labels


def test_reactivity_mesh1():
    # Test basics

    # Prepare

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    # Render once
    render(cube)

    # Changing the color should not change anything
    cube.material.color = "red"
    assert render(cube) == set()

    # Changing the render mask requires new render info
    cube.render_mask = "all"
    assert render(cube) == {"render"}

    # Changing the side requires a new pipeline
    cube.material.side = "FRONT"
    assert render(cube) == {"pipeline"}

    # Changing the wireframe requires a new shader
    cube.material.wireframe = True
    assert render(cube) == {"shader"}


def test_reactivity_mesh2():
    # Test swapping resources

    # Prepare

    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

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

    # Render once
    render(cube)

    # Swap out the material - same type
    cube.material = m2
    assert render(cube) == {"resources"}

    # Swap out the material - different type
    cube.material = m3
    assert render(cube) == {"pipeline", "shader", "resources", "render"}

    # Swap out the positions
    cube.geometry.positions = p2
    assert render(cube) == {"resources"}

    # Swap out the whole geometry
    cube.geometry = g2
    assert render(cube) == {"resources"}


def test_reactivity_mesh3():
    # Test swapping colormap

    geometry = gfx.torus_knot_geometry(1, 0.3, 128, 32)
    geometry.texcoords = gfx.Buffer(geometry.texcoords.data[:, 0])

    tex1 = gfx.cm.cividis
    tex2 = gfx.cm.inferno
    cmap3 = np.array([(1,), (0,), (0,), (1,)], np.int32)
    tex3 = gfx.Texture(cmap3, dim=1).get_view(filter="linear")

    obj = gfx.Mesh(geometry, gfx.MeshPhongMaterial(map=tex1))

    # Render once
    render(obj)

    # Change to a colormap with the same format, all is ok!
    obj.material.map = tex2
    assert render(obj) == {"resources"}

    # Change to colormap of different format, need rebuild!
    obj.material.map = tex3
    assert render(obj) != {"resources"}


if __name__ == "__main__":
    for ob in list(globals().values()):
        if callable(ob) and ob.__name__.startswith("test_"):
            print(ob.__name__)
            ob()
    print("done")
