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


if __name__ == "__main__":
    test_uniform_types_uniform_type()
