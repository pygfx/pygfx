from pygfx.renderers.wgpu.shader.resolve import resolve_includes

codes = {}
codes[
    "a.wgsl"
] = """
    fn a() -> f32{
        return 1.0;
    }
"""

codes[
    "b.wgsl"
] = """
    fn b() -> f32{
        return 1.0;
    }
"""

codes[
    "c.wgsl"
] = """
    #include a.wgsl
    fn c() -> f32{
        return 1.0;
    }
"""

codes[
    "d.wgsl"
] = """
    #include a.wgsl
    #include b.wgsl
    #include c.wgsl
    fn d() -> f32{
        return 1.0;
    }
"""


def load_func(uri):
    return codes[uri]


def test_includes_none():
    code = """"
    fn main() {
    }
    """

    ref = code

    result = resolve_includes(code, load_func)
    assert result == ref


def test_includes_simple():
    code = """
    // bla
    #include a.wgsl

    fn main() {
    }
    """

    ref = """
    fn a() -> f32{
        return 1.0;
    }

    // bla
    // #include a.wgsl

    fn main() {
    }
    """

    result = resolve_includes(code, load_func)
    assert result == ref


def test_includes_simple():
    code = """
    // bla
    #include a.wgsl

    fn main() {
    }
    """

    ref = """
    fn a() -> f32{
        return 1.0;
    }

    // bla
    // #include a.wgsl

    fn main() {
    }
    """

    result = resolve_includes(code, load_func)
    assert result == ref


def test_includes_recursive():
    code = """
    #include b.wgsl
    #include c.wgsl

    fn main() {
    }
    """

    ref = """
    fn b() -> f32{
        return 1.0;
    }

    fn a() -> f32{
        return 1.0;
    }

    // #include a.wgsl
    fn c() -> f32{
        return 1.0;
    }

    // #include b.wgsl
    // #include c.wgsl

    fn main() {
    }
    """

    result = resolve_includes(code, load_func)
    assert result == ref



def test_includes_recursive_with_dups():
    code = """
    #include a.wgsl
    #include c.wgsl

    fn main() -> f32{
    }
    """

    ref = """
    fn a() -> f32{
        return 1.0;
    }

    // #include a.wgsl
    fn c() -> f32{
        return 1.0;
    }

    // #include a.wgsl
    // #include c.wgsl

    fn main() -> f32{
    }
    """

    result = resolve_includes(code, load_func)
    assert result == ref


def test_includes_more_recursive():
    code = "#include d.wgsl"

    ref = """
    fn a() -> f32{
        return 1.0;
    }

    fn b() -> f32{
        return 1.0;
    }

    // #include a.wgsl
    fn c() -> f32{
        return 1.0;
    }

    // #include a.wgsl
    // #include b.wgsl
    // #include c.wgsl
    fn d() -> f32{
        return 1.0;
    }
// #include d.wgsl"""

    result = resolve_includes(code, load_func)
    assert result == ref

if __name__ == "__main__":
    test_includes_none()
    test_includes_simple()
    test_includes_recursive()
    test_includes_recursive_with_dups()
    test_includes_more_recursive()