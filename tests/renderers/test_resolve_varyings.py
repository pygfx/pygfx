from pygfx.renderers.wgpu.shader.resolve import resolve_varyings
from pytest import raises


def test_varying_basics():
    assert resolve_varyings("") == ""
    assert resolve_varyings("foo") == "foo\n"
    assert resolve_varyings("\nfoo\n") == "foo\n"
    assert resolve_varyings("\n\nfoo\n\n") == "foo\n"
    assert resolve_varyings("\n\n    foo\n\n") == "    foo\n"


def test_varyings_struct_position0():
    # If no varyings are used, the struct is not inserted

    code1 = """
    @vertex
    fn vs_main() {
    }
    fn fs_main() {
    }
    """

    code2 = code1

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct_position1():
    # But if the struct type is used, we insert it

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct_position2():
    # Indentation is preserved

    code1 = """
            @vertex
            fn vs_main() -> Varyings {
            }
            fn fs_main(varyings : Varyings) {
            }
    """

    code2 = """
            struct Varyings {
            };

            @vertex
            fn vs_main() -> Varyings {
            }
            fn fs_main(varyings : Varyings) {
            }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct_position3():
    # Struct is positioned before first function that uses it

    code1 = """
    fn foo() = {
    }
    @vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    fn foo() = {
    }
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct_position4():
    # Or at the start if no function was detected

    code1 = """
    let x = 3;
    var y : Varyings;
    """

    code2 = """
    struct Varyings {
    };

    let x = 3;
    var y : Varyings;
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct_position5():
    # Positioning keeps entrypoint stuff into account

    code1 = """
    fn foo() = {
    }
    @@vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    fn foo() = {
    }
    struct Varyings {
    };

    @@vertex
    fn vs_main() -> Varyings {
    }
    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_remove1():
    # If no varyings are used, any set varyings are removed.

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        // unused: varyings.foo = f32(something1);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_remove2():
    # Used varyings are held

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.bar);
    }
    """

    code2 = """
    struct Varyings {
        @location(0) bar : vec2<f32>,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        // unused: varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        // unused: varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.bar);
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_remove3():
    # Comments are taken into account

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        // use(varyings.bar, varyings.foo);
        // let x = varyings.spam;
    }
    """

    code2 = """
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        // unused: varyings.foo = f32(something1);
        // unused: varyings.bar = vec2<f32>(something2);
        // unused: varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        // use(varyings.bar, varyings.foo);
        // let x = varyings.spam;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_remove4():
    # Usage can actually be to the left of the assignment op

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        something[i32(varyings.foo)] = 0.0;
    }
    """

    code2 = """
    struct Varyings {
        @location(0) foo : f32,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        something[i32(varyings.foo)] = 0.0;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_remove5():
    # Varying assignments can be multi-line

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.spam = vec3<f32>(
            1.0, 2.0, 3.0
        );
        return varyings;
    }
    """

    code2 = """
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        // unused: varyings.spam = vec3<f32>(
        // 1.0, 2.0, 3.0
        // );
        return varyings;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_struct1():
    # Usage in other funcs counts, and varyings are sorted alphabetically

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.foo,varyings.bar);
    }

    fn func(varyings: Varyings) {
        let x = varyings.spam + 2.0;
    }
    """

    code2 = """
    struct Varyings {
        @location(0) bar : vec2<f32>,
        @location(1) foo : f32,
        @location(2) spam : vec3<f32>,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        varyings.spam = vec3<f32>(something3);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.foo,varyings.bar);
    }

    fn func(varyings: Varyings) {
        let x = varyings.spam + 2.0;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_attr1():
    # Can set varying attribute

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.color = vec4<f32>(color);
        varyings.color.a = 1.0;
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        varyings.color;
    }
    """

    code2 = """
    struct Varyings {
        @location(0) color : vec4<f32>,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.color = vec4<f32>(color);
        varyings.color.a = 1.0;
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        varyings.color;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_attr2():
    # Can set varying attribute, and all setter lines are commented

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.color = vec4<f32>(color);
        varyings.color.a = 1.0;
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    struct Varyings {
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        // unused: varyings.color = vec4<f32>(color);
        // unused: varyings.color.a = 1.0;
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_builtin1():
    # Builtins are not slots

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.foo,varyings.bar);
        let pos = varyings.position;
    }
    """

    code2 = """
    struct Varyings {
        @location(0) bar : vec2<f32>,
        @location(1) foo : f32,
        @builtin(position) position : vec4<f32>,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        varyings.foo = f32(something1);
        varyings.bar = vec2<f32>(something2);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
        use(varyings.foo,varyings.bar);
        let pos = varyings.position;
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_builtin2():
    # Builtins are considered used when set

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code2 = """
    struct Varyings {
        @builtin(position) position : vec4<f32>,
    };

    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    code3 = resolve_varyings(code1)
    assert code3.strip() == code2.strip()


def test_varyings_error_readonly_in_vs1():
    # Cannot use varying as a getter in vs_main

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        varyings.ndc = vec3<f32>(varyings.position.xyz);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    with raises(TypeError) as info:
        resolve_varyings(code1)
    info.match("only writing is allowed")


def test_varyings_error_need_type_when_set():
    # Assignment needs type annotation

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = ndc_pos;
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    with raises(TypeError) as info:
        resolve_varyings(code1)
    info.match("type")


def test_varyings_error_assignments_invalid_multiline1():
    # Assignment needs type annotation

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = vec4<f32>(
    """.strip()

    with raises(TypeError) as info:
        resolve_varyings(code1)
    info.match("missing a semicolon")

    code = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.foo = vec4<f32>(
            1.0, 2.0, 3.0, 4.0
        )
        XX
    """

    for xx in ("", "}", "let y = 1;", "var : f32;"):
        code1 = code.replace("XX", xx)
        with raises(TypeError) as info:
            resolve_varyings(code1)
        info.match("missing a semicolon")


def test_varyings_error_need_type_match():
    # When assigning multiple times, the type should match

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
        var varyings : Varyings;
        varyings.position = vec4<f32>(ndc_pos);
        varyings.position = vec3<f32>(ndc_pos);
        return varyings;
    }

    fn fs_main(varyings : Varyings) {
    }
    """

    with raises(TypeError) as info:
        resolve_varyings(code1)
    info.match("expected type")


def test_varyings_error_used_but_not_assigned():
    # When assigning multiple times, the type should match

    code1 = """
    @vertex
    fn vs_main() -> Varyings {
    }

    fn fs_main(varyings : Varyings) {
        varyings.foo
    }
    """

    with raises(TypeError) as info:
        resolve_varyings(code1)
    info.match("not assigned")


if __name__ == "__main__":
    for f in list(globals().values()):
        if callable(f) and f.__name__.startswith("test_"):
            print(f.__name__)
            f()
