import wgpu
from ._update import update_buffer
from ._utils import to_vertex_format

from ...objects import PointLight

# todo: idea:
# the shader and a corresponding binding layout for vertex data can also be defined
# on the Shader object. That way, line and point objects can also casts shadows :)


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


class ShadowUtil:
    def __init__(self, device):
        self.device = device
        self.pipelines = {}
        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=binding_layout
        )

    def get_shadow_pipeline(self, wobject):

        # shadow pipeline only depends on the object's geometry info (Vertex Buffer)
        # TODO: now it only depends on the positions, but for instanced meshes, it also needs instance buffer
        #  Maybe for future skinned meshes , morph buffer is also needed
        positions = wobject.geometry.positions

        array_stride = positions.nbytes // positions.nitems
        vertex_format = to_vertex_format(positions.format)

        hash = (array_stride, vertex_format)

        if hash not in self.pipelines:
            self.pipelines[hash] = self._create_shadow_pipeline(
                array_stride, vertex_format
            )

        return self.pipelines[hash]

    def _create_shadow_pipeline(self, array_stride, vertex_format):

        vertex_buffer_descriptor = [
            {
                "array_stride": array_stride,
                "step_mode": wgpu.VertexStepMode.vertex,  # vertex
                "attributes": [
                    {
                        "format": vertex_format,
                        "offset": 0,
                        "shader_location": 0,
                    }
                ],
            }
        ]

        program = self.device.create_shader_module(code=shadow_vertex_shader)
        pipeline = self.device.create_render_pipeline(
            layout=self.device.create_pipeline_layout(
                bind_group_layouts=[self.bind_group_layout, self.bind_group_layout]
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

    def render_shadow_maps(self, lights, wobjects, command_encoder):
        """Render the shadow map for the given lights and wobjects."""

        for light in lights:
            if light.cast_shadow:

                if isinstance(light, PointLight):
                    for buffer in light.shadow._gfx_matrix_buffer:
                        update_buffer(self.device, buffer)
                else:
                    update_buffer(self.device, light.shadow._gfx_matrix_buffer)

                if not isinstance(light.shadow._wgpu_tex_view, list):
                    shadow_maps = [light.shadow._wgpu_tex_view]
                else:
                    shadow_maps = light.shadow._wgpu_tex_view

                for i, shadow_map in enumerate(shadow_maps):
                    shadow_pass = command_encoder.begin_render_pass(
                        color_attachments=[],
                        depth_stencil_attachment={
                            "view": shadow_map,
                            "depth_clear_value": 1.0,
                            "depth_load_op": wgpu.LoadOp.clear,
                            "depth_store_op": wgpu.StoreOp.store,
                            "stencil_load_op": wgpu.LoadOp.clear,
                            "stencil_store_op": wgpu.StoreOp.store,
                        },
                    )

                    if not hasattr(light.shadow, f"__shadow_bind_group_{i}"):

                        buffer = (
                            light.shadow._gfx_matrix_buffer[i]._wgpu_buffer[1]
                            if isinstance(light, PointLight)
                            else light.shadow._gfx_matrix_buffer._wgpu_buffer[1]
                        )

                        setattr(
                            light.shadow,
                            f"__shadow_bind_group_{i}",
                            self.device.create_bind_group(
                                layout=self.bind_group_layout,
                                entries=[
                                    {
                                        "binding": 0,
                                        "resource": {
                                            "buffer": buffer,
                                            "offset": 0,
                                            "size": 64,
                                        },
                                    }
                                ],
                            ),
                        )

                    shadow_pass.set_bind_group(
                        0, getattr(light.shadow, f"__shadow_bind_group_{i}"), [], 0, 99
                    )

                    for wobject in wobjects:
                        if wobject.cast_shadow and wobject.geometry is not None:
                            shadow_pipeline = self.get_shadow_pipeline(wobject)

                            if shadow_pipeline is not None:
                                shadow_pass.set_pipeline(shadow_pipeline)

                                position_buffer = wobject.geometry.positions

                                shadow_pass.set_vertex_buffer(
                                    0,
                                    position_buffer._wgpu_buffer[1],
                                    position_buffer.vertex_byte_range[0],
                                    position_buffer.vertex_byte_range[1],
                                )

                                if not hasattr(wobject, f"__shadow_bind_group"):
                                    # Note that here we assume that the wobject's transform matrix
                                    # is the first item in its uniform buffer. This is a likely
                                    # assumption because items are sorted by size.
                                    bg = self.device.create_bind_group(
                                        layout=self.bind_group_layout,
                                        entries=[
                                            {
                                                "binding": 0,
                                                "resource": {
                                                    "buffer": wobject.uniform_buffer._wgpu_buffer[
                                                        1
                                                    ],
                                                    "offset": 0,
                                                    "size": 64,
                                                },
                                            }
                                        ],
                                    )

                                    setattr(wobject, "__shadow_bind_group", bg)

                                shadow_pass.set_bind_group(
                                    1,
                                    getattr(wobject, "__shadow_bind_group"),
                                    [],
                                    0,
                                    99,
                                )

                                ibuffer = wobject.geometry.indices

                                n = wobject.geometry.indices.data.size
                                n_instance = 1  # not support instanced meshes yet

                                if ibuffer is not None:
                                    index_format = to_vertex_format(ibuffer.format)
                                    index_format = index_format.split("x")[0].replace(
                                        "s", "u"
                                    )
                                    shadow_pass.set_index_buffer(
                                        ibuffer._wgpu_buffer[1], index_format
                                    )
                                    shadow_pass.draw_indexed(n, n_instance)
                                else:
                                    shadow_pass.draw(n, n_instance)

                    shadow_pass.end()
