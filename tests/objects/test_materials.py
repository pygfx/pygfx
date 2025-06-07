import pygfx


# Collect materials
material_classes = []
for name in dir(pygfx.materials):
    val = getattr(pygfx.materials, name)
    if isinstance(val, type) and issubclass(val, pygfx.Material):
        material_classes.append(val)


def test_uniform_types_uniform_type():
    """Check that the uniform_type always includes that of the super."""
    for cls in material_classes:
        ob = cls()
        assert isinstance(ob.uniform_type, dict)
        for super_cls in cls.mro():
            if super_cls is cls:
                continue
            elif not hasattr(super_cls, "uniform_type"):
                break
            else:
                for key, val in super_cls.uniform_type.items():
                    assert key in ob.uniform_type, f"{cls.__name__}:{key} missing"
                    assert val == ob.uniform_type[key], (
                        f"{cls.__name__}:{key} different"
                    )


def test_automatic_props():
    m = pygfx.Material()

    # Default case
    assert not m.transparent_is_set
    assert not m.depth_write_is_set
    assert m.transparent is None
    assert m.depth_write is True

    # Use opacity
    m.opacity = 0.5
    assert not m.transparent_is_set
    assert not m.depth_write_is_set
    assert m.transparent is True
    assert m.depth_write is False

    # Use transparent True
    m.opacity = 1
    m.transparent = True
    assert m.transparent_is_set
    assert not m.depth_write_is_set
    assert m.transparent is True
    assert m.depth_write is False

    # Use transparent False
    m.opacity = 1
    m.transparent = False
    assert m.transparent_is_set
    assert not m.depth_write_is_set
    assert m.transparent is False
    assert m.depth_write is True

    # Use depth_write
    m.opacity = 1
    m.transparent = None
    m.depth_write = False
    assert not m.transparent_is_set
    assert m.depth_write_is_set
    assert m.transparent is None
    assert m.depth_write is False


if __name__ == "__main__":
    test_uniform_types_uniform_type()
    test_automatic_props()
