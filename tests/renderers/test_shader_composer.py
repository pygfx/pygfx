from pygfx.renderers.wgpu import _shadercomposer as shadercomposer
from pytest import raises
import numpy as np


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
        shader.define_uniform(0, 0, "zz", "not a struct")
    with raises(TypeError):  # Not a valid struct type
        shader.define_uniform(0, 0, "zz", np.array([1]).dtype)

    # Test simple scalars
    struct = dict(foo="f4", bar="i2")
    shader.define_uniform(0, 0, "zz", struct)
    assert (
        shader.get_definitions().strip()
        == """
        [[block]]
        struct Struct_zz {
            foo: f32;
            bar: i16;
        };

        [[group(0), binding(0)]]
        var<uniform> zz: Struct_zz;
    """.strip()
    )

    # Test vec
    struct = dict(foo="4xf4", bar="2xi4")
    shader.define_uniform(0, 0, "zz", struct)
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
    struct = dict(foo="4x4xf4", bar="2x3xi4")
    shader.define_uniform(0, 0, "zz", struct)
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

    # Test alignment
    struct = dict(foo="3xf4", bar="4xi4")
    with raises(TypeError):
        shader.define_uniform(0, 0, "zz", struct)
    struct = dict(foo="3xf4", _padding="f4", bar="4xi4")
    shader.define_uniform(0, 0, "zz", struct)


if __name__ == "__main__":
    test_templating()
    test_uniform_definitions()
