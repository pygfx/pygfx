from pygfx.utils import array_from_shadertype


def test_array_from_shadertype_scalar():
    # A simple dict with two floats
    d = dict(
        foo="f4",
        bar="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar")
    assert a.nbytes == 8  # 4 + 4

    # Order is preserved
    d = dict(
        bar="f4",
        foo="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("bar", "foo")
    assert a.nbytes == 8  # 4 + 4


def test_array_from_shadertype_vec2():
    # A 2-vector and a float
    d = dict(
        foo="2xf4",
        bar="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar", "__padding1")
    assert a.nbytes == 16  # 8 + 4 + padding to 8

    # Order is based on alignment
    d = dict(
        bar="f4",
        foo="2xf4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar", "__padding1")
    assert a.nbytes == 16  # 8 + 4 + padding to 8


def test_array_from_shadertype_vec3():
    # A 3-vector has alignment 16
    d = dict(
        foo="3xf4",
        bar="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar")
    assert a.nbytes == 16  # 12 + 4

    # The float will be hoisted up to fill the gap.
    # In this case it does not make a difference in the total size
    d = dict(
        foo1="3xf4",
        foo2="3xf4",
        bar="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo1", "bar", "foo2", "__padding1")
    assert a.nbytes == 32  # 12 + 12 + 4 + padding to 16

    # But in this case it does: no padding!
    d = dict(
        foo1="3xf4",
        foo2="3xf4",
        bar1="f4",
        bar2="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo1", "bar1", "foo2", "bar2")
    assert a.nbytes == 32  # 12 + 12 + 4 + 4

    # With nothing to fill it up, uses padding
    d = dict(
        foo1="3xf4",
        foo2="3xf4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo1", "__padding1", "foo2", "__padding2")
    assert a.nbytes == 32  # 12 + 4 + 12 + 4


def test_array_from_shadertype_mat2():
    # A nx2-mat has alignment 8
    d = dict(
        foo="3x2xf4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo",)
    assert a.nbytes == 24  # padded to 8

    # A nx2-mat has alignment 8
    d = dict(
        foo="3x2xf4",
        bar="f4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar", "__padding1")
    assert a.nbytes == 32  #  + 24 + 4 + padding to 8


def test_array_from_shadertype_mat3():
    # A nx3-mat has alignment 16.
    d = dict(
        foo="2x3xf4",
    )
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "__meta_mat3_names__foo")
    assert a.nbytes == 32  # 24 + pad to 16

    # These also *needs* internal padding.
    # See the SizeOf in https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
    # So that bar-field cannot be used to fill the gap, like we did for vec3.
    d = dict(foo="3x3xf4", bar="f4")
    a = array_from_shadertype(d)
    assert a.dtype.names == ("foo", "bar", "__padding1", "__meta_mat3_names__foo")
    assert a.nbytes == 64  # 36 + 12 internal padding + 4 + pad to 16


if __name__ == "__main__":
    test_array_from_shadertype_scalar()
    test_array_from_shadertype_vec2()
    test_array_from_shadertype_vec3()
    test_array_from_shadertype_mat2()
    test_array_from_shadertype_mat3()
