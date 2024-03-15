import pygfx as gfx
from pytest import raises


class Object1(gfx.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


class Object2(gfx.WorldObject):
    def __init__(self, material):
        super().__init__()
        self.material = material


class Object3(Object1):
    pass


class Material1(gfx.Material):
    pass


class Material2(gfx.Material):
    pass


class Material3(Material1):
    pass


def foo1():
    pass


def foo2():
    pass


def foo3():
    pass


def test_render_registry_api():
    assert gfx.renderers.wgpu.register_wgpu_render_function
    assert gfx.renderers.svg.register_svg_render_function


def test_render_registry_fails():
    r = gfx.utils.renderfunctionregistry.RenderFunctionRegistry()

    # This is ok
    r.register(Object1, Material1, foo1)

    # Type errors for first arg
    with raises(TypeError):
        r.register(4, Material1, foo1)
    with raises(TypeError):
        r.register(str, Material1, foo1)
    # Type errors for second arg
    with raises(TypeError):
        r.register(Object1, 4, foo1)
    with raises(TypeError):
        r.register(Object1, str, foo1)
    # Type errors for third arg
    with raises(TypeError):
        r.register(Object1, Material1, "not callable")

    # Cannot register with the same types twice
    with raises(ValueError):
        r.register(Object1, Material1, foo1)

    assert len(r._store) == 1

    # This is ok
    assert foo1 is r.get_render_function(Object1(Material1()))

    # Material of None (or missing) is allowed, but returns None
    assert None is r.get_render_function(Object1(None))

    # Given object (and its material) must be the right type
    with raises(TypeError):
        r.get_render_function(3)
    with raises(TypeError):
        r.get_render_function(Object1(3))


def test_render_registry_selection():
    r = gfx.utils.renderfunctionregistry.RenderFunctionRegistry()

    # Register one combo
    r.register(Object1, Material1, foo1)

    assert foo1 is r.get_render_function(Object1(Material1()))
    assert None is r.get_render_function(Object1(Material2()))
    assert foo1 is r.get_render_function(Object1(Material3()))

    assert None is r.get_render_function(Object2(Material1()))
    assert None is r.get_render_function(Object2(Material2()))
    assert None is r.get_render_function(Object2(Material3()))

    assert foo1 is r.get_render_function(Object3(Material1()))
    assert None is r.get_render_function(Object3(Material2()))
    assert foo1 is r.get_render_function(Object3(Material3()))

    # Register another
    r.register(Object2, Material2, foo2)

    assert foo1 is r.get_render_function(Object1(Material1()))
    assert None is r.get_render_function(Object1(Material2()))
    assert foo1 is r.get_render_function(Object1(Material3()))

    assert None is r.get_render_function(Object2(Material1()))
    assert foo2 is r.get_render_function(Object2(Material2()))
    assert None is r.get_render_function(Object2(Material3()))

    assert foo1 is r.get_render_function(Object3(Material1()))
    assert None is r.get_render_function(Object3(Material2()))
    assert foo1 is r.get_render_function(Object3(Material3()))

    # Register another two
    r.register(Object3, Material1, foo3)
    r.register(Object3, Material2, foo3)
    # r.register(Object3, Material3, foo3) -> not necessary

    assert foo1 is r.get_render_function(Object1(Material1()))
    assert None is r.get_render_function(Object1(Material2()))
    assert foo1 is r.get_render_function(Object1(Material3()))

    assert None is r.get_render_function(Object2(Material1()))
    assert foo2 is r.get_render_function(Object2(Material2()))
    assert None is r.get_render_function(Object2(Material3()))

    assert foo3 is r.get_render_function(Object3(Material1()))
    assert foo3 is r.get_render_function(Object3(Material2()))
    assert foo3 is r.get_render_function(Object3(Material3()))

    # Now make foo1 and foo2 a catch all
    r.register(Object1, gfx.Material, foo1)
    r.register(Object2, gfx.Material, foo2)

    assert foo1 is r.get_render_function(Object1(Material1()))
    assert foo1 is r.get_render_function(Object1(Material2()))
    assert foo1 is r.get_render_function(Object1(Material3()))

    assert foo2 is r.get_render_function(Object2(Material1()))
    assert foo2 is r.get_render_function(Object2(Material2()))
    assert foo2 is r.get_render_function(Object2(Material3()))

    assert foo3 is r.get_render_function(Object3(Material1()))
    assert foo3 is r.get_render_function(Object3(Material2()))
    assert foo3 is r.get_render_function(Object3(Material3()))


def test_render_registry_of_wgpu():
    r = gfx.renderers.wgpu.engine.utils.registry

    assert None is r.get_render_function(Object1(Material1()))

    assert r.get_render_function(gfx.Mesh(None, gfx.MeshBasicMaterial()))


def test_render_registry_of_svg():
    r = gfx.renderers.svg._svgrenderer.registry

    assert None is r.get_render_function(Object1(Material1()))

    assert r.get_render_function(gfx.Line(None, gfx.LineMaterial()))


if __name__ == "__main__":
    test_render_registry_api()
    test_render_registry_fails()
    test_render_registry_selection()
    test_render_registry_of_wgpu()
    test_render_registry_of_svg()
