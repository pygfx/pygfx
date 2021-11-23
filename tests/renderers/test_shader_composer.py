from pygfx.renderers.wgpu import _shadercomposer as shadercomposer
from pytest import raises
import numpy as np


Binding = shadercomposer.Binding


def test_templating():
    class MyShader(shadercomposer.BaseShader):
        def get_code(self):
            return """
            $$ if foo
            x = {{bar}}
            $$ else
            x = {{bar + 1}}
            $$ endif
            """

    # Missing variables
    shader = MyShader(foo=True)
    with raises(ValueError):
        shader.generate_wgsl()

    # Fill in value
    shader["bar"] = 42
    assert shader["bar"] == 42
    assert shader.generate_wgsl().strip() == "x = 42"

    # Can also specify when generating
    assert shader.generate_wgsl(foo=False).strip() == "x = 43"

    # Inline block notation
    class MyShader(shadercomposer.BaseShader):
        def get_code(self):
            return """
            {$ if foo $} 1 {$ else $} 2 {$ endif $}
            """

    shader = MyShader(foo=True)
    assert shader.generate_wgsl().strip() == "1"
    assert shader.generate_wgsl(foo=False).strip() == "2"


def test_uniform_definitions():
    class MyShader(shadercomposer.BaseShader):
        def get_code(self):
            return ""

    shader = MyShader()

    # Fails
    with raises(TypeError):  # Not a valid struct type
        shader.define_binding(0, 0, Binding("zz", "buffer/uniform", "not a struct"))
    with raises(TypeError):  # Not a valid struct type
        shader.define_binding(
            0, 0, Binding("zz", "buffer/uniform", np.array([1]).dtype)
        )

    # Test simple scalars
    struct = dict(foo="f4", bar="i4")
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        shader.get_definitions().strip()
        == """
        [[block]]
        struct Struct_zz {
            foo: f32;
            bar: i32;
        };

        [[group(0), binding(0)]]
        var<uniform> zz: Struct_zz;
    """.strip()
    )

    # Test vec
    struct = dict(foo="4xf4", bar="2xi4")
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        shader.get_definitions().strip()
        == """
        [[block]]
        struct Struct_zz {
            foo: vec4<f32>;
            bar: vec2<i32>;
        };

        [[group(0), binding(0)]]
        var<uniform> zz: Struct_zz;
    """.strip()
    )

    # Test mat
    struct = dict(foo="4x4xf4", bar="3x2xi4")
    shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))
    assert (
        shader.get_definitions().strip()
        == """
        [[block]]
        struct Struct_zz {
            foo: mat4x4<f32>;
            bar: mat3x2<i32>;
        };

        [[group(0), binding(0)]]
        var<uniform> zz: Struct_zz;
    """.strip()
    )

    # Test that it forbids types that align badly.
    # There are two cases where the alignment is checked.
    # In array_from_shadertype() the fields are ordered to prevent
    # alignment mismatches, and it will prevent the use of some types.
    # Later we might implement introducing stub fields to fix alignment.
    # In define_binding() the uniform's wgsl struct definition is created,
    # and there it also checks the alignment. Basically a failsafe for
    # when array_from_shadertype() does not do it's job right.
    struct = dict(foo="3xf4", bar="4xi4")
    with raises(ValueError):
        shader.define_binding(0, 0, Binding("zz", "buffer/uniform", struct))


def test_resolve_depth_output():
    resolve_depth_output = shadercomposer.resolve_depth_output

    code1 = """
    struct FragmentOutput {
        [[location(0)]] color: vec4<f32>;
    }
    fn fs_main() {
    }
    """.strip()
    code2 = code1
    assert resolve_depth_output(code1).strip() == code2

    code1 = """
    struct FragmentOutput {
        [[location(0)]] color: vec4<f32>;
    }
    fn fs_main() {
        out.depth = 0.0;
    }
    """

    code2 = """
    struct FragmentOutput {
        [[builtin(frag_depth)]] depth : f32;
        [[location(0)]] color: vec4<f32>;
    }
    fn fs_main() {
        out.depth = 0.0;
    }
    """.strip()

    assert resolve_depth_output(code1).strip() == code2


if __name__ == "__main__":
    test_templating()
    test_uniform_definitions()
    test_resolve_depth_output()
