import wgpu

from .... import objects
from .update import ensure_wgpu_object, update_resource
from .utils import to_index_format, to_vertex_format, GpuCache
from .shared import get_shared


# This cache enables re-using gpu pipelines for calculating shadows,
# these can be shared between multiple world-objects that have a
# positions buffer with a matching stride and format.
SHADOW_CACHE = GpuCache("shadow_pipelines")


shadow_vertex_shader = """
    struct Matrix4x4 {
        m : mat4x4<f32>
    };
    @group(0) @binding(0) var<uniform> light_view_projection : Matrix4x4;
    @group(1) @binding(0) var<uniform> model_transform : Matrix4x4;
    @vertex
    fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
        return light_view_projection.m * model_transform.m * vec4<f32>(position, 1.0);
    }
"""

instanced_shadow_vertex_shader = """
    struct Matrix4x4 {
        m : mat4x4<f32>
    };

    struct InstanceInput {
        @location(1) model_matrix_0: vec4<f32>,
        @location(2) model_matrix_1: vec4<f32>,
        @location(3) model_matrix_2: vec4<f32>,
        @location(4) model_matrix_3: vec4<f32>,
    };

    @group(0) @binding(0) var<uniform> light_view_projection : Matrix4x4;
    @group(1) @binding(0) var<uniform> model_transform : Matrix4x4;
    @vertex
    fn vs_main(@location(0) position: vec3<f32>, instance: InstanceInput) -> @builtin(position) vec4<f32> {
        let instance_matrix = mat4x4<f32>(
            instance.model_matrix_0,
            instance.model_matrix_1,
            instance.model_matrix_2,
            instance.model_matrix_3
        );
        return light_view_projection.m * (model_transform.m * instance_matrix) * vec4<f32>(position, 1.0);
    }
"""

bind_group_layout_entry = {
    "binding": 0,
    "visibility": wgpu.ShaderStage.VERTEX,
    "buffer": {"type": wgpu.BufferBindingType.uniform},
}


global_bind_group_layout = None


def render_shadow_maps(lights, wobjects, command_encoder):
    """Render the wobjects into the shadow maps for the given lights."""

    global global_bind_group_layout

    # Make sure the global bind_group_layout is ready
    device = get_shared().device
    if global_bind_group_layout is None:
        global_bind_group_layout = device.create_bind_group_layout(
            entries=[bind_group_layout_entry]
        )

    # Filter shadow-able objects once beforehand.
    wobjects = [w for w in wobjects if w.cast_shadow and w.geometry is not None]

    for light in lights:
        if not light.cast_shadow:
            continue

        # What kind of light is this?
        light_has_6_sides = isinstance(light, objects.PointLight)
        assert light_has_6_sides == isinstance(light.shadow._wgpu_tex_view, list)
        assert light_has_6_sides == isinstance(light.shadow._gfx_matrix_buffer, list)

        if light_has_6_sides:
            # Render each shadow map
            shadow_maps = light.shadow._wgpu_tex_view
            shadow_buffers = light.shadow._gfx_matrix_buffer
            for i in range(6):
                render_shadow_map(
                    device,
                    light,
                    shadow_maps[i],
                    shadow_buffers[i],
                    wobjects,
                    command_encoder,
                )
        else:
            # Render this one shadow map
            shadow_map = light.shadow._wgpu_tex_view
            shadow_buffer = light.shadow._gfx_matrix_buffer
            render_shadow_map(
                device, light, shadow_map, shadow_buffer, wobjects, command_encoder
            )


def render_shadow_map(
    device, light, shadow_map, shadow_buffer, wobjects, command_encoder
):
    """Render the wobjects into the given shadow map."""

    shadow_pass = command_encoder.begin_render_pass(
        color_attachments=[],
        depth_stencil_attachment={
            "view": shadow_map,
            "depth_read_only": False,
            "depth_clear_value": 1.0,
            "depth_load_op": wgpu.LoadOp.clear,
            "depth_store_op": wgpu.StoreOp.store,
            "stencil_read_only": True,
            "stencil_load_op": wgpu.LoadOp.clear,
            "stencil_store_op": wgpu.StoreOp.discard,
        },
    )

    light_bind_group = get_shadow_bind_group(device, shadow_buffer)
    shadow_pass.set_bind_group(0, light_bind_group, [], 0, 99)

    for wobject in wobjects:
        render_wobject_shadow(device, light, wobject, shadow_pass)

    shadow_pass.end()


def render_wobject_shadow(device, light, wobject, shadow_pass):
    """Render one wobject into a shadow map."""

    shadow_pipeline = get_shadow_pipeline(device, wobject, light.shadow.cull_mode)
    shadow_pass.set_pipeline(shadow_pipeline)

    position_buffer = wobject.geometry.positions
    shadow_pass.set_vertex_buffer(
        0,
        ensure_wgpu_object(position_buffer),
        position_buffer.draw_range[0] * position_buffer.itemsize,
        position_buffer.draw_range[1] * position_buffer.itemsize,
    )

    n_instance = 1

    if isinstance(wobject, objects.InstancedMesh):
        instance_buffer = wobject.instance_buffer
        shadow_pass.set_vertex_buffer(1, ensure_wgpu_object(instance_buffer))
        n_instance = instance_buffer.nitems

    wobject_bind_group = get_shadow_bind_group(device, wobject.uniform_buffer)
    shadow_pass.set_bind_group(1, wobject_bind_group, [], 0, 99)

    ibuffer = getattr(wobject.geometry, "indices", None)

    if ibuffer is not None:
        n = wobject.geometry.indices.data.size
        shadow_pass.set_index_buffer(
            ensure_wgpu_object(ibuffer), to_index_format(ibuffer.format)
        )
        shadow_pass.draw_indexed(n, n_instance)
    else:
        n = wobject.geometry.positions.nitems
        shadow_pass.draw(n, n_instance)


def get_shadow_bind_group(device, shadow_buffer):
    """Get the bind group object for this shadow buffer."""

    # Since the bind-group is bound one-to-one to the buffer, there is
    # no need for sophisticated caching:  The bind group is cached on
    # the buffer itself.

    bind_group = getattr(shadow_buffer, "_gfx_shadow_bind_group", None)

    if bind_group is None:
        shadow_buffer._wgpu_usage |= wgpu.BufferUsage.UNIFORM
        wgpu_buffer = ensure_wgpu_object(shadow_buffer)
        # We also update the buffer now, because this code gets called
        # *after* the renderer has synced all resources. Note that this
        # code is only reached once for each shadow map.
        update_resource(shadow_buffer)

        bind_group = device.create_bind_group(
            layout=global_bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": wgpu_buffer,
                        "offset": 0,
                        "size": 64,
                    },
                }
            ],
        )

        shadow_buffer._gfx_shadow_bind_group = bind_group

    return bind_group


def get_shadow_pipeline(device, wobject, cull_mode):
    """Get the pipeline object for rendering a wobject into a shadow map."""

    # The shadow pipeline only depends on the object's geometry info
    # (Vertex Buffer), so there is potential for sharing this pipeline
    # between objects. We use a proper cache to implement this.
    #
    # NOTE:
    # - for instanced meshes, it also needs instance buffer.
    # - maybe for future skinned meshes, the morph buffer is also needed.

    position_buffer = wobject.geometry.positions

    stride = position_buffer.itemsize
    format = position_buffer.format
    topology = get_shadow_topology(wobject)

    is_instanced = isinstance(wobject, objects.InstancedMesh)

    key = (stride, format, topology, cull_mode, is_instanced)

    pipeline = SHADOW_CACHE.get(key)
    if pipeline is None:
        # Create pipeline and store in the cache (with a weakref).
        pipeline = create_shadow_pipeline(
            device, stride, format, topology, cull_mode, is_instanced
        )
        SHADOW_CACHE.set(key, pipeline)

    # Store on the wobject to bind it to its lifetime, but per shadow-cull-mode
    setattr(wobject, f"_gfx_shadow_pipeline_{cull_mode}", pipeline)
    return pipeline


def get_shadow_topology(wobject):
    """Get the primitive topology for drawing the shadows of this object."""
    if isinstance(wobject, objects.Mesh):
        return wgpu.PrimitiveTopology.triangle_list
    elif isinstance(wobject, objects.Line):
        return wgpu.PrimitiveTopology.line_strip
    elif isinstance(wobject, objects.Points):
        return wgpu.PrimitiveTopology.point_list
    else:
        raise RuntimeError(f"Shadows not supported for {wobject.__class__.__name__}")


def create_shadow_pipeline(
    device, stride, format, topology, cull_mode, instanced=False
):
    """Actually create a shadow pipeline object."""

    vertex_buffer_descriptor = [
        {
            "array_stride": stride,
            "step_mode": wgpu.VertexStepMode.vertex,  # vertex
            "attributes": [
                {
                    "format": to_vertex_format(format),
                    "offset": 0,
                    "shader_location": 0,
                }
            ],
        }
    ]

    if instanced:
        vertex_buffer_descriptor.append(
            {
                "array_stride": 80,  # matrix4x4(64) + id(4) + padding(12) = 80
                "step_mode": wgpu.VertexStepMode.instance,  # instance
                "attributes": [
                    {
                        "format": "float32x4",
                        "offset": 0,
                        "shader_location": 1,
                    },
                    {
                        "format": "float32x4",
                        "offset": 16,
                        "shader_location": 2,
                    },
                    {
                        "format": "float32x4",
                        "offset": 32,
                        "shader_location": 3,
                    },
                    {
                        "format": "float32x4",
                        "offset": 48,
                        "shader_location": 4,
                    },
                ],
            }
        )
        shader_module = device.create_shader_module(code=instanced_shadow_vertex_shader)
    else:
        shader_module = device.create_shader_module(code=shadow_vertex_shader)

    pipeline = device.create_render_pipeline(
        layout=device.create_pipeline_layout(
            bind_group_layouts=[global_bind_group_layout, global_bind_group_layout]
        ),
        vertex={
            "module": shader_module,
            "entry_point": "vs_main",
            "buffers": vertex_buffer_descriptor,
        },
        primitive={
            "topology": topology,
            "cull_mode": cull_mode.lower(),
        },
        depth_stencil={
            "format": wgpu.TextureFormat.depth32float,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
            "stencil_read_mask": 0,
            "stencil_write_mask": 0,
        },
    )

    return pipeline
