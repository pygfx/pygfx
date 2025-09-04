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
    m = pygfx.Material(alpha_mode="auto")

    # Default case for 'auto'
    assert not m.depth_write_is_set
    assert m.depth_write is True
    assert m.render_queue == 2600
    assert m.depth_write is True

    # Go transparent
    m.alpha_mode = "blend"
    assert m.render_queue == 3000
    assert m.depth_write is False

    # Go opaque
    m.alpha_mode = "solid"
    assert m.render_queue == 2000
    assert m.depth_write is True

    # With alpha test
    m.alpha_mode = "solid"
    m.alpha_test = 0.5
    assert m.render_queue == 2400
    assert m.depth_write is True

    # Restore
    m.alpha_test = 0
    assert m.render_queue == 2000
    assert m.depth_write is True

    # Go Stochastic
    m.alpha_mode = "dither"
    assert m.render_queue == 2400
    assert m.depth_write is True

    # Overriding
    m.alpha_mode = "auto"

    # Set to "transparent" queue
    m.render_queue = 3000
    assert m.render_queue_is_set
    assert m.render_queue == 3000
    assert m.depth_write is True

    # Set to "opaque" queue
    m.render_queue = 2000
    assert m.render_queue_is_set
    assert m.render_queue == 2000
    assert m.depth_write is True

    # Restore
    m.render_queue = None
    assert not m.render_queue_is_set

    # Use depth_write
    m.depth_write = False
    assert m.depth_write_is_set
    assert m.render_queue == 2600
    assert m.depth_write is False


if __name__ == "__main__":
    test_uniform_types_uniform_type()
    test_automatic_props()
