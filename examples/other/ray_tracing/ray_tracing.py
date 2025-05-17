"""
Ray Tracing
===========

This example demonstrates a simple ray tracing implementation using wgpu and pygfx.
It renders a Cornell box scene with a few objects and a light source.
"""

# sphinx_gallery_pygfx_docs = 'code'
# sphinx_gallery_pygfx_test = 'off'

import wgpu
import numpy as np
import math
from wgpu.utils.imgui import Stats
from wgpu.gui.auto import WgpuCanvas, run
from pathlib import Path
from camera import OrbitCamera
from scene import Material, parse_gfx_scene
import pygfx as gfx
import pylinalg as la


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

COMPUTE_WORKGROUP_SIZE_X = 8
COMPUTE_WORKGROUP_SIZE_Y = 8
MAX_BOUNCES = 50


def create_scene():
    gfx_scene = gfx.Scene()

    red = Material((0.63, 0.065, 0.05), roughness=1.0)
    green = Material((0.14, 0.45, 0.091), roughness=1.0)
    white = Material((0.725, 0.71, 0.68), roughness=1.0)

    light_color1 = np.array([0.747 + 0.058, 0.747 + 0.285, 0.747])
    light_color2 = np.array([0.740 + 0.287, 0.740 + 0.160, 0.740])
    light_color3 = np.array([0.737 + 0.642, 0.737 + 0.159, 0.737])
    light_color = 8.0 * light_color1 + 15.6 * light_color2 + 18.4 * light_color3
    light_color = light_color / math.pi

    light = Material((0.78, 0.78, 0.78), emissive=light_color, roughness=1.0)

    specular = Material((1.0, 1.0, 1.0), roughness=0.04, metallic=1.0)
    glass = Material((1.0, 1.0, 1.0), roughness=0.0, metallic=0.0, ior=1.5)

    # Cornell box scene

    ROOM_SIZE = 55.5  # noqa
    BOX_SIZE = 16.5  # noqa
    HALF_ROOM_SIZE = ROOM_SIZE / 2  # noqa

    wall_geometry = gfx.plane_geometry(width=ROOM_SIZE, height=ROOM_SIZE)

    # left wall
    left_wall_mesh = gfx.Mesh(wall_geometry, red)
    left_wall_mesh.local.position = (-HALF_ROOM_SIZE, 0, 0)
    left_wall_mesh.local.rotation = la.quat_from_euler((0, math.pi / 2, 0))
    gfx_scene.add(left_wall_mesh)

    # right wall
    right_wall_mesh = gfx.Mesh(wall_geometry, green)
    right_wall_mesh.local.position = (HALF_ROOM_SIZE, 0, 0)
    right_wall_mesh.local.rotation = la.quat_from_euler((0, math.pi / 2, 0))
    gfx_scene.add(right_wall_mesh)

    # floor
    floor_mesh = gfx.Mesh(wall_geometry, white)
    floor_mesh.local.position = (0, -HALF_ROOM_SIZE, 0)
    floor_mesh.local.rotation = la.quat_from_euler((math.pi / 2, 0, 0))
    gfx_scene.add(floor_mesh)

    # ceiling
    ceiling_mesh = gfx.Mesh(wall_geometry, white)
    ceiling_mesh.local.position = (0, HALF_ROOM_SIZE, 0)
    ceiling_mesh.local.rotation = la.quat_from_euler((math.pi / 2, 0, 0))
    gfx_scene.add(ceiling_mesh)

    # front wall
    front_wall_mesh = gfx.Mesh(wall_geometry, white)
    front_wall_mesh.local.position = (0, 0, HALF_ROOM_SIZE)
    front_wall_mesh.local.rotation = la.quat_from_euler((0, 0, 0))
    gfx_scene.add(front_wall_mesh)

    # light
    light_g = gfx.plane_geometry(width=13, height=10.5)
    light_mesh = gfx.Mesh(light_g, light)
    light_mesh.local.position = (0, HALF_ROOM_SIZE - 0.2, 0)
    light_mesh.local.rotation = la.quat_from_euler((math.pi / 2, 0, 0))
    gfx_scene.add(light_mesh)

    box1_group = gfx.Group()

    box1_left = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE * 2)
    box1_left_mesh = gfx.Mesh(box1_left, white)
    box1_left_mesh.local.rotation = la.quat_from_euler((0, math.pi / 2, 0))
    box1_left_mesh.local.position = (-BOX_SIZE / 2, 0, 0)

    box1_right = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE * 2)
    box1_right_mesh = gfx.Mesh(box1_right, white)
    box1_right_mesh.local.rotation = la.quat_from_euler((0, math.pi / 2, 0))
    box1_right_mesh.local.position = (BOX_SIZE / 2, 0, 0)

    box1_back = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE * 2)
    box1_back_mesh = gfx.Mesh(box1_back, white)
    box1_back_mesh.local.rotation = la.quat_from_euler((0, 0, 0))
    box1_back_mesh.local.position = (0, 0, BOX_SIZE / 2)

    box1_top = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE)
    box1_top_mesh = gfx.Mesh(box1_top, white)
    box1_top_mesh.local.rotation = la.quat_from_euler((math.pi / 2, 0, 0))
    box1_top_mesh.local.position = (0, BOX_SIZE, 0)

    box1_bottom = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE)
    box1_bottom_mesh = gfx.Mesh(box1_bottom, white)
    box1_bottom_mesh.local.rotation = la.quat_from_euler((math.pi / 2, 0, 0))
    box1_bottom_mesh.local.position = (0, -BOX_SIZE, 0)

    box1_front = gfx.plane_geometry(width=BOX_SIZE, height=BOX_SIZE * 2)
    box1_front_mesh = gfx.Mesh(box1_front, specular)
    box1_front_mesh.local.rotation = la.quat_from_euler((0, 0, 0))
    box1_front_mesh.local.position = (0, 0, -BOX_SIZE / 2)

    box1_group.add(box1_left_mesh)
    box1_group.add(box1_right_mesh)
    box1_group.add(box1_back_mesh)
    box1_group.add(box1_top_mesh)
    box1_group.add(box1_bottom_mesh)
    box1_group.add(box1_front_mesh)

    off_set = (BOX_SIZE - ROOM_SIZE) / 2
    box1_group.local.position = (
        -off_set - 26.5 - 2.0,
        BOX_SIZE - HALF_ROOM_SIZE,
        29.5 + off_set,
    )
    box1_group.local.rotation = la.quat_from_euler((0, -15 / 180 * math.pi, 0))
    gfx_scene.add(box1_group)

    # box1 = gfx.box_geometry(width=BOX_SIZE, height=BOX_SIZE*2, depth=BOX_SIZE)
    # box_mesh1 = gfx.Mesh(box1, white)
    # off_set = (BOX_SIZE - ROOM_SIZE) / 2
    # box_mesh1.local.position = (-off_set-26.5 -2.0, BOX_SIZE - HALF_ROOM_SIZE, 29.5+off_set)
    # box_mesh1.local.rotation = la.quat_from_euler((0, -15 / 180 * math.pi, 0))
    # gfx_scene.add(box_mesh1)

    box2 = gfx.box_geometry(width=BOX_SIZE, height=BOX_SIZE, depth=BOX_SIZE)
    box_mesh2 = gfx.Mesh(box2, glass)
    off_set = (BOX_SIZE - ROOM_SIZE) / 2
    box_mesh2.local.position = (-off_set - 13.0 + 2.0, off_set, 6.5 + off_set)
    box_mesh2.local.rotation = la.quat_from_euler((0, 18 / 180 * math.pi, 0))
    gfx_scene.add(box_mesh2)

    triangles, materials = parse_gfx_scene(gfx_scene)

    return triangles, materials


def load_wgsl(shader_name):
    shader_dir = Path(__file__).parent / "shaders"
    shader_path = shader_dir / shader_name
    with open(shader_path, "rb") as f:
        return f.read().decode()


canvas = WgpuCanvas(
    title="Raytracing", size=(IMAGE_WIDTH, IMAGE_HEIGHT), max_fps=-1, vsync=False
)

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(
    required_features=["texture-adapter-specific-format-features", "float32-filterable"]
)
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)


frame_texture = device.create_texture(
    size=(IMAGE_WIDTH, IMAGE_HEIGHT, 1),
    usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    format=wgpu.TextureFormat.rgba32float,
    dimension="2d",
)

triangles, materials = create_scene()


triangles_buffer = device.create_buffer_with_data(
    data=triangles.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)

materials_buffer = device.create_buffer_with_data(
    data=materials.tobytes(),
    usage=wgpu.BufferUsage.STORAGE,
)

common_buffer_data = np.zeros(
    (),
    dtype=[
        ("width", "uint32"),
        ("height", "uint32"),
        ("frame_counter", "uint32"),
        ("__padding2", "uint32"),  # padding to 16 bytes
    ],
)


common_buffer = device.create_buffer(
    size=common_buffer_data.nbytes,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

camera = OrbitCamera(
    canvas=canvas,
    fov=40.0,
    focal_length=20.0,
    defocus_angle=0.0,
    center=(0.0, 0.0, 0.0),
    distance=110.0,
    # attitude=math.pi / 24,
    azimuth=math.pi,
    up=(0.0, 1.0, 0.0),
)

camera_buffer_data = camera.buffer_data

camera_buffer = device.create_buffer_with_data(
    data=camera_buffer_data.tobytes(),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

# Create raytracing compute shader

UTIL = load_wgsl("util.wgsl")
MATERIAL = load_wgsl("material.wgsl")
RAY = load_wgsl("ray.wgsl")
COMPUTE_SHADER = load_wgsl("ray_tracing.wgsl")
CAMERA = load_wgsl("camera.wgsl")

ray_tracing_src = "\n".join([UTIL, MATERIAL, RAY, CAMERA, COMPUTE_SHADER])


ray_tracing_pipeline = device.create_compute_pipeline(
    layout="auto",
    compute={
        "module": device.create_shader_module(code=ray_tracing_src),
        "entry_point": "main",
        "constants": {
            "WORKGROUP_SIZE_X": COMPUTE_WORKGROUP_SIZE_X,
            "WORKGROUP_SIZE_Y": COMPUTE_WORKGROUP_SIZE_Y,
            "OBJECTS_COUNT_IN_SCENE": len(triangles),
            "MAX_BOUNCES": MAX_BOUNCES,
        },
    },
)


ray_tracing_bind_group = device.create_bind_group(
    layout=ray_tracing_pipeline.get_bind_group_layout(0),
    entries=[
        {"binding": 0, "resource": {"buffer": triangles_buffer}},
        {"binding": 1, "resource": {"buffer": materials_buffer}},
        {"binding": 2, "resource": {"buffer": common_buffer}},
        {"binding": 3, "resource": {"buffer": camera_buffer}},
        {"binding": 4, "resource": frame_texture.create_view()},
    ],
)


# Render to the screen
RENDER_SHADER = load_wgsl("render.wgsl")

render_src = "\n".join([RENDER_SHADER])

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
        {"binding": 0, "resource": sampler},
        {"binding": 1, "resource": frame_texture.create_view()},
    ],
)

state = {"spp": 0}


def do_ray_tracing():
    canvas_texture = present_context.get_current_texture()

    # Update camera buffer
    if camera._need_calculate_buffer:
        camera.calculate_buffer_data()

    if camera._need_update_buffer:
        device.queue.write_buffer(camera_buffer, 0, camera_buffer_data.tobytes())
        common_buffer_data["frame_counter"] = 0
        camera._need_update_buffer = False

    # Update uniform buffer
    common_buffer_data["width"] = IMAGE_WIDTH
    common_buffer_data["height"] = IMAGE_HEIGHT
    common_buffer_data["frame_counter"] += 1
    state["spp"] = common_buffer_data["frame_counter"]

    device.queue.write_buffer(common_buffer, 0, common_buffer_data.tobytes())

    command_encoder = device.create_command_encoder()

    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(ray_tracing_pipeline)
    compute_pass.set_bind_group(0, ray_tracing_bind_group)
    compute_pass.dispatch_workgroups(
        math.ceil(IMAGE_WIDTH / COMPUTE_WORKGROUP_SIZE_X),
        math.ceil(IMAGE_HEIGHT / COMPUTE_WORKGROUP_SIZE_Y),
        1,
    )
    compute_pass.end()

    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": canvas_texture.create_view(),
                "load_op": "clear",
                "store_op": "store",
                "clear_value": (0.0, 0.0, 0.0, 1.0),
            }
        ],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_bind_group(0, render_bind_group)
    render_pass.draw(3, 1, 0, 0)
    render_pass.end()

    device.queue.submit([command_encoder.finish()])


stats = Stats(device, canvas)

stats.extra_data = state


def loop():
    with stats:
        do_ray_tracing()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(loop)
    run()
