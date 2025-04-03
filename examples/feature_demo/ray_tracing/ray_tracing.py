import wgpu
import pygfx as gfx
import numpy as np
import math
from wgpu.gui.auto import WgpuCanvas, run
from pathlib import Path


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

COMPUTE_WORKGROUP_SIZE_X = 8
COMPUTE_WORKGROUP_SIZE_Y = 8


class Sphere(np.ndarray):
    def __new__(cls):
        return np.zeros(1, dtype=np.dtype([
            ('center', np.float32, 3),
            ('radius', np.float32),
            ('albedo', np.float32, 3),
            ('material_type', np.int32),
        ])).view(cls)

def create_scene():
    spheres = []
    
    ground = Sphere()
    ground['center'] = [0, -100.5, -2.0]
    ground['radius'] = 100.0
    ground['albedo'] = [0.8, 0.8, 0.8]
    ground['material_type'] = 0  # 0=漫反射
    spheres.append(ground)
    
    materials = [
        ([0.8, 0.3, 0.3], 0),    # 漫反射红
        ([0.8, 0.6, 0.2], 1),    # 金属
        ([0.8, 0.8, 0.8], 2),    # 玻璃
    ]
    for i, (albedo, mat_type) in enumerate(materials):
        sphere = Sphere()
        sphere['center'] = [i * 2.0 - 2.0, 0.0, -2.0]
        sphere['radius'] = 0.5
        sphere['albedo'] = albedo
        sphere['material_type'] = mat_type
        spheres.append(sphere)
    
    return np.concatenate(spheres)


def load_wgsl(shader_name):
    shader_dir = Path(__file__).parent / "shaders"
    shader_path = shader_dir / shader_name
    with open(shader_path, "rb") as f:
        return f.read().decode()


canvas = WgpuCanvas(title="WebGPU Raytracing", size=(IMAGE_WIDTH, IMAGE_HEIGHT))

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(required_features=["texture-adapter-specific-format-features"])
present_context = canvas.get_context()
# render_texture_format = present_context.get_preferred_format(device.adapter)
render_texture_format = wgpu.TextureFormat.rgba8unorm
present_context.configure(device=device, format=render_texture_format)

# Create a buffer for the ray tracing result
# frame_buffer = device.create_buffer(
#     size=IMAGE_WIDTH * IMAGE_HEIGHT * 4 * 4,  # f32x4,
#     usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
# )

frame_texture = device.create_texture(
    size=(IMAGE_WIDTH, IMAGE_HEIGHT, 1),
    usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    format=render_texture_format,
    dimension="2d",
)

spheres = create_scene()
sphere_buffer = device.create_buffer_with_data(
    data=spheres.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)

# random seed buffer
rng_state= np.arange(0, IMAGE_WIDTH*IMAGE_HEIGHT, dtype=np.uint32)
rng_state_buffer = device.create_buffer_with_data(
    data=rng_state.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)


common_buffer_data = np.zeros(
    (),
    dtype=[
        ("viewport_size", "uint32", (2)),
        ("frame_counter", "uint32"),
        ("__padding", "uint32"),  # padding to 16 bytes
    ],
)

common_buffer_data["frame_counter"] = 0

common_buffer = device.create_buffer(
    size=common_buffer_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

#Create raytracing compute shader

COMMON = load_wgsl("common.wgsl")
UTIL = load_wgsl("util.wgsl")
RAY = load_wgsl("ray.wgsl")
COMPUTE_SHADER = load_wgsl("ray_tracing.wgsl")
CAMERA = load_wgsl("camera.wgsl")

ray_tracing_src = "\n".join([COMMON, UTIL, RAY, CAMERA, COMPUTE_SHADER])


ray_tracing_pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={
        "module": device.create_shader_module(code=ray_tracing_src),
        "entry_point": "main",
        "constants": {
            "WORKGROUP_SIZE_X": COMPUTE_WORKGROUP_SIZE_X,
            "WORKGROUP_SIZE_Y": COMPUTE_WORKGROUP_SIZE_Y,
            "OBJECTS_COUNT_IN_SCENE": len(spheres),
        },
    
    },
)


ray_tracing_bind_group = device.create_bind_group(
    layout=ray_tracing_pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": sphere_buffer}},
        # {"binding": 1, "resource": {"buffer": frame_buffer}},
        {"binding": 1, "resource": frame_texture.create_view()},
        {"binding": 2, "resource": {"buffer": rng_state_buffer}},   
        {"binding": 3, "resource": {"buffer": common_buffer}},
    ],
)


# Render to the screen
RENDER_SHADER = load_wgsl("render.wgsl")

render_src = "\n".join([COMMON, RENDER_SHADER])

render_module = device.create_shader_module(
    code=render_src,
)

render_pipeline = device.create_render_pipeline(
    layout="auto",
    vertex={
        "module": render_module,
        "entry_point": "vs_main",
        "buffers": [],
    },
    fragment={
        "module": render_module,
        "entry_point": "fs_main",
        "targets": [
            {
                "format": render_texture_format,
            }
        ],
    },
)


sampler = device.create_sampler(
    mag_filter="linear",
    min_filter="linear",
    mipmap_filter="linear",
)

render_bind_group = device.create_bind_group(
    layout=render_pipeline.get_bind_group_layout(0),
    entries=[
        # {"binding": 0, "resource": {"buffer": common_buffer}},
        # {"binding": 1, "resource": {"buffer": frame_buffer}},
        {"binding": 0, "resource": sampler},
        {"binding": 1, "resource": frame_texture.create_view()},
    ],
)

def on_draw():

    canvas_texture = present_context.get_current_texture()

    # Update uniform buffer
    common_buffer_data["viewport_size"] = (canvas_texture.size[0], canvas_texture.size[1])
    common_buffer_data["frame_counter"] += 1

    device.queue.write_buffer(common_buffer, 0, common_buffer_data.tobytes())


    command_encoder = device.create_command_encoder()
    
    # 执行计算着色器`   `
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(ray_tracing_pipeline)
    compute_pass.set_bind_group(0, ray_tracing_bind_group)
    compute_pass.dispatch_workgroups( math.ceil(IMAGE_WIDTH / COMPUTE_WORKGROUP_SIZE_X), math.ceil(IMAGE_HEIGHT / COMPUTE_WORKGROUP_SIZE_Y), 1)
    compute_pass.end()


    # 执行渲染着色器

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[{
            "view": canvas_texture.create_view(),
            "load_op": "clear",
            "store_op": "store",
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, render_bind_group)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()


    device.queue.submit([command_encoder.finish()])

    canvas.request_draw()

if __name__ == "__main__":
    canvas.request_draw(on_draw)
    run()