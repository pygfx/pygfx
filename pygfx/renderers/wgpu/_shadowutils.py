import wgpu


shadow_vertex_shader = """

struct Matrix4x4 {
    matrix : mat4x4<f32>
};

@group(0) @binding(0) var<uniform> light_view_projection : Matrix4x4;
@group(1) @binding(0) var<uniform> model_transform : Matrix4x4;

@stage(vertex)
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light_view_projection.matrix * model_transform.matrix * vec4<f32>(position, 1.0);
}

"""

binding_layout = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.VERTEX,
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    }
]


def get_shadow_bind_group_layout(shared):
    if "shadow_bgl" not in shared.shader_cache:
        bind_group_layout = shared.device.create_bind_group_layout(
            entries=binding_layout
        )

        shared.shader_cache["shadow_bgl"] = bind_group_layout
    return shared.shader_cache["shadow_bgl"]


def get_shadow_program(shared):
    if "shadow_shader" not in shared.shader_cache:
        program = shared.device.create_shader_module(code=shadow_vertex_shader)
        shared.shader_cache["shadow_shader"] = program
    return shared.shader_cache["shadow_shader"]


# TODO can be cached
def create_shadow_pipeline(shared, vertex_buffer_descriptor):

    device = shared.device

    bind_group_layout = get_shadow_bind_group_layout(shared)
    program = get_shadow_program(shared)

    pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout, bind_group_layout]
        ),
        vertex={
            "module": program,
            "entry_point": "vs_main",
            "buffers": vertex_buffer_descriptor,
        },
        primitive={
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "cull_mode": "front",
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth32float,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        },
    )

    return pipeline
