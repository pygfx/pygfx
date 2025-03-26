from pygfx.renderers.wgpu.shader.resolve import resolve_output


def test_fragmentoutput_nothing():
    # No fields are set, and there are no virtual fields.
    code1 = """
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
    }
    """.strip()
    code2 = code1
    assert resolve_output(code1).strip() == code2


def test_fragmentoutput_no_virtual_fields():
    # No virtual fields, but the real fields are set
    code1 = """
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
        @location(1) foo: f32,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.color = vec4<f32>(1.0);
        out.foo = 42.0;
        return out;
    }
    """.strip()
    code2 = code1
    assert resolve_output(code1).strip() == code2


def test_fragmentoutput_no_output():
    # Has virtual fields, but code does not use FragmentOutput in fragment shader
    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    fn some function() {
        var out: FragmentOutput;
        out.color = vec4<f32>(1.0);
        out.foo = 42.0;
        return out;
    }
    """.strip()
    code2 = code1
    assert resolve_output(code1).strip() == code2


def test_fragmentoutput_virtual_using_default():
    # out.foo is not specified, so it uses the default

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        apply_virtual_fields_of_fragment_output(&out, 42);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_no_return():
    # test special case; does not rely on return out

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        apply_virtual_fields_of_fragment_output(&out, 42);
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_is_set():
    # out.foo is set, use that

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.foo = 7;
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_foo: f32;
        out_virtualfield_foo = 7;
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_foo);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_is_set_in_branch():
    # out.foo is set, use that

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        if (something) {
            out.foo = 7;
        } else {
            out.foo = 9;
        }
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_foo: f32;
        if (something) {
            out_virtualfield_foo = 7;
        } else {
            out_virtualfield_foo = 9;
        }
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_foo);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_is_also_real_field_unset():
    # out.color is both a real and a virtual field, but not set

    code1 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        apply_virtual_fields_of_fragment_output(&out, vec4<f32>(0.0));
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_is_also_real_field_set():
    # out.color is both a real and a virtual field, and set

    code1 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.color = some_color;
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_color: vec4<f32>;
        out_virtualfield_color = some_color;
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_color);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_virtual_is_multiline():
    # out.foo is set using a multiline value

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : vec2<f32> = vec2<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.foo = vec2<f32>(
                1.0,
                2.0
        );
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : vec2<f32> = vec2<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_foo: vec2<f32>;
        out_virtualfield_foo = vec2<f32>(
                1.0,
                2.0
        );
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_foo);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_multiple_places():
    # Capable of dealing with multiple end-points

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        // virtualfield bar : vec2<f32> = vec2<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main1() {
        var out: FragmentOutput;
        out.foo = 7;
        return out;
    }
    fn another function() {
        let out = 2;
        return out;
    }
    @fragment
    fn fs_main2() {
        var out: FragmentOutput;
        out.bar = vec2<f32>(1.0, 2.0);
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        // virtualfield bar : vec2<f32> = vec2<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main1() {
        var out: FragmentOutput;
        var out_virtualfield_foo: f32;
        out_virtualfield_foo = 7;
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_foo, vec2<f32>(0.0));
        return out;
    }
    fn another function() {
        let out = 2;
        return out;
    }
    @fragment
    fn fs_main2() {
        var out: FragmentOutput;
        var out_virtualfield_bar: vec2<f32>;
        out_virtualfield_bar = vec2<f32>(1.0, 2.0);
        apply_virtual_fields_of_fragment_output(&out, 42, out_virtualfield_bar);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_add_depth_simple():
    # depth field is set

    code1 = """
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.depth = 0.0;
        return out;
    }
    """

    code2 = """
    struct FragmentOutput {
        @builtin(frag_depth) depth : f32,
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.depth = 0.0;
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_add_depth_with_virtual():
    # Depth field is set, and there are also virtual fields

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.color = vec4<f32>(1.0);
        out.depth = 0.0;
        return out;
    }
    """

    code2 = """
    struct FragmentOutput {
        @builtin(frag_depth) depth : f32,
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.color = vec4<f32>(1.0);
        out.depth = 0.0;
        apply_virtual_fields_of_fragment_output(&out, 42);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_legacy_simple():
    # using get_fragment_output(), simple

    code1 = """
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out = get_fragment_output(varyings.position, out_color);
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        out.color = get_fragment_output(varyings.position, out_color).color;
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_legacy_with_virtual():
    # using get_fragment_output(), and a virtual field

    code1 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out = get_fragment_output(varyings.position, out_color);
        out.foo = 7;
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield foo : f32 = 42;
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_foo: f32;
        out.color = get_fragment_output(varyings.position, out_color).color;
        out_virtualfield_foo = 7;
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_foo);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


def test_fragmentoutput_legacy_virtual_is_real():
    # using get_fragment_output(), and using color as both a virtual and real field

    code1 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out = get_fragment_output(varyings.position, out_color);
        return out;
    }
    """.strip()

    code2 = """
    struct FragmentOutput {
        // virtualfield color : vec4<f32> = vec4<f32>(0.0);
        @location(0) color: vec4<f32>,
    }
    @fragment
    fn fs_main() {
        var out: FragmentOutput;
        var out_virtualfield_color: vec4<f32>;
        out_virtualfield_color = get_fragment_output(varyings.position, out_color).color;
        apply_virtual_fields_of_fragment_output(&out, out_virtualfield_color);
        return out;
    }
    """.strip()

    code3 = resolve_output(code1).strip()
    assert code3 == code2


if __name__ == "__main__":
    for f in list(globals().values()):
        if callable(f) and f.__name__.startswith("test_"):
            print(f.__name__)
            f()
