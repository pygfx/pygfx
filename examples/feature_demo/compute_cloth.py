"""
Cloth Simulation with Compute Shaders
=====================================

This example demonstrates how to implement a Verlet integration cloth physics simulation using compute shaders,
including a spring system, gravity, wind forces, and sphere collisions.
"""

# sphinx_gallery_pygfx_docs = 'animate 4s'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import pygfx as gfx
import wgpu
import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop
from pygfx.utils.compute import ComputeShader
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiRenderer
from wgpu.utils.imgui import Stats

canvas = RenderCanvas(
    size=(1280, 720), update_mode="fastest", title="compute cloth", vsync=False
)
renderer = gfx.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(45, 1280 / 720, depth_range=(0.01, 10))

scene = gfx.Scene()

# cloth parameters
cloth_width = 1.0
cloth_height = 1.0
cloth_segments_x = 30
cloth_segments_y = 30
sphere_radius = 0.15

# Verlet system parameters
verlet_vertices = []
verlet_springs = []
vertex_columns = []


def setup_verlet_geometry():
    global verlet_vertices, verlet_springs

    vertex_id = 0

    # create Verlet vertices
    for x in range(cloth_segments_x + 1):
        column = []
        for y in range(cloth_segments_y + 1):
            pos_x = x * (cloth_width / cloth_segments_x) - cloth_width * 0.5
            pos_z = y * (cloth_height / cloth_segments_y)
            pos_y = cloth_height * 0.5

            # fixed vertices at the top
            is_fixed = (y == 0) and (x % 5 == 0)

            vertex = {
                "id": vertex_id,
                "position": np.array([pos_x, pos_y, pos_z], dtype=np.float32),
                "is_fixed": is_fixed,
                "spring_ids": [],
            }
            verlet_vertices.append(vertex)
            column.append(vertex)
            vertex_id += 1

        vertex_columns.append(column)

    # create Verlet springs
    spring_id = 0

    def add_verlet_spring(vertex0, vertex1):
        nonlocal spring_id
        spring = {
            "id": spring_id,
            "vertex0_id": vertex0["id"],
            "vertex1_id": vertex1["id"],
            "rest_length": np.linalg.norm(vertex0["position"] - vertex1["position"]),
        }
        vertex0["spring_ids"].append(spring_id)
        vertex1["spring_ids"].append(spring_id)
        verlet_springs.append(spring)
        spring_id += 1

    for x in range(cloth_segments_x + 1):
        for y in range(cloth_segments_y + 1):
            vertex0 = vertex_columns[x][y]
            if x > 0:
                add_verlet_spring(vertex0, vertex_columns[x - 1][y])
            if y > 0:
                add_verlet_spring(vertex0, vertex_columns[x][y - 1])
            if x > 0 and y > 0:
                add_verlet_spring(vertex0, vertex_columns[x - 1][y - 1])
            if x > 0 and y < cloth_segments_y:
                add_verlet_spring(vertex0, vertex_columns[x - 1][y + 1])


def setup_verlet_buffers():
    vertex_count = len(verlet_vertices)
    spring_count = len(verlet_springs)

    # Verlet vertex positions buffer
    positions = np.array([v["position"] for v in verlet_vertices], dtype=np.float32)
    position_buffer = gfx.Buffer(
        data=positions,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.VERTEX,
    )

    # Verlet vertex forces buffer
    forces = np.zeros((vertex_count, 3), dtype=np.float32)
    force_buffer = gfx.Buffer(
        data=forces, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    # Verlet vertex params buffer (is_fixed, spring_count, spring_pointer)
    spring_list = []
    vertex_params = np.zeros((vertex_count, 3), dtype=np.uint32)

    for i, vertex in enumerate(verlet_vertices):
        vertex_params[i, 0] = 1 if vertex["is_fixed"] else 0
        vertex_params[i, 1] = len(vertex["spring_ids"])
        vertex_params[i, 2] = len(spring_list)
        spring_list.extend(vertex["spring_ids"])

    vertex_params_buffer = gfx.Buffer(
        data=vertex_params, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    # spring list buffer
    spring_list_buffer = gfx.Buffer(
        data=np.array(spring_list, dtype=np.uint32),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    # spring data buffers
    spring_vertex_ids = np.array(
        [[s["vertex0_id"], s["vertex1_id"]] for s in verlet_springs], dtype=np.uint32
    )
    spring_rest_lengths = np.array(
        [s["rest_length"] for s in verlet_springs], dtype=np.float32
    )

    spring_vertex_ids_buffer = gfx.Buffer(
        data=spring_vertex_ids,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    spring_rest_lengths_buffer = gfx.Buffer(
        data=spring_rest_lengths,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    # spring forces buffer
    spring_forces = np.zeros((spring_count, 3), dtype=np.float32)
    spring_force_buffer = gfx.Buffer(
        data=spring_forces, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    # param buffer [dampening, sphere_pos(3), sphere_active, wind, stiffness, time]
    params_data = np.array([0.99, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.0], dtype=np.float32)
    params_buffer = gfx.Buffer(
        data=params_data, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    return (
        position_buffer,
        force_buffer,
        vertex_params_buffer,
        spring_list_buffer,
        spring_vertex_ids_buffer,
        spring_rest_lengths_buffer,
        spring_force_buffer,
        params_buffer,
        vertex_count,
        spring_count,
    )


# setup Verlet geometry and buffers
setup_verlet_geometry()

(
    position_buffer,
    force_buffer,
    vertex_params_buffer,
    spring_list_buffer,
    spring_vertex_ids_buffer,
    spring_rest_lengths_buffer,
    spring_force_buffer,
    params_buffer,
    vertex_count,
    spring_count,
) = setup_verlet_buffers()

# todo: now, storage buffers (gfx.Buffer) are not supported array<vec3<f32>>, so we use array<f32> instead.

# compute shaders for Verlet integration
spring_compute_wgsl = """
struct Params {
    dampening: f32,
    sphere_x: f32,
    sphere_y: f32,
    sphere_z: f32,
    sphere_active: f32,
    wind: f32,
    stiffness: f32,
    time: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> spring_vertex_ids: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> spring_rest_lengths: array<f32>;
@group(0) @binding(4) var<storage, read_write> spring_forces: array<f32>;

@compute @workgroup_size(64)
fn compute_spring_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&spring_rest_lengths)) {
        return;
    }

    let vertex_ids = spring_vertex_ids[index];
    let rest_length = spring_rest_lengths[index];

    // let vertex0_pos = positions[vertex_ids.x];
    let vertex0_pos = vec3<f32>(positions[vertex_ids.x * 3u], positions[vertex_ids.x * 3u + 1u], positions[vertex_ids.x * 3u + 2u]);
    // let vertex1_pos = positions[vertex_ids.y];
    let vertex1_pos = vec3<f32>(positions[vertex_ids.y * 3u], positions[vertex_ids.y * 3u + 1u], positions[vertex_ids.y * 3u + 2u]);

    let delta = vertex1_pos - vertex0_pos;
    let dist = length(delta);
    let force = (dist - rest_length) * params.stiffness * delta * 0.5 / dist;

    // spring_forces[index] = force;
    spring_forces[index * 3u] = force.x;
    spring_forces[index * 3u + 1u] = force.y;
    spring_forces[index * 3u + 2u] = force.z;
}

"""

vertex_compute_wgsl = """
struct Params {
    dampening: f32,
    sphere_x: f32,
    sphere_y: f32,
    sphere_z: f32,
    sphere_active: f32,
    wind: f32,
    stiffness: f32,
    time: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces: array<f32>;
@group(0) @binding(3) var<storage, read> vertex_params: array<u32>;
@group(0) @binding(4) var<storage, read> spring_list: array<u32>;
@group(0) @binding(5) var<storage, read> spring_vertex_ids: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> spring_forces: array<f32>;

override SPHERE_RADIUS: f32 = 0.15;

fn tri(x: f32) -> f32 {
    return abs(fract(x) - 0.5);
}

fn tri3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        tri(p.z + tri(p.y * 1.0)),
        tri(p.z + tri(p.x * 1.0)),
        tri(p.y + tri(p.x * 1.0))
    );
}

fn triNoise3D(position: vec3<f32>, speed: f32, time: f32) -> f32 {
    var p = position;
    var z = 1.4;
    var rz = 0.0;
    var bp = position;

    for (var i = 0.0; i <= 3.0; i += 1.0) {
        let dg = tri3(bp * 2.0);
        p += dg + time * 0.1 * speed;
        bp *= 1.8;
        z *= 1.5;
        p *= 1.2;

        let t = tri(p.z + tri(p.x + tri(p.y)));
        rz += t / z;
        bp += vec3<f32>(0.14);
    }

    return rz;
}

@compute @workgroup_size(64)
fn compute_vertex_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&positions)/3u) {
        return;
    }

    // let vertex_param = vertex_params[index];
    let vertex_param = vec3<u32>(vertex_params[index * 3u], vertex_params[index * 3u + 1u], vertex_params[index * 3u + 2u]);
    let is_fixed = vertex_param.x;
    let spring_count = vertex_param.y;
    let spring_pointer = vertex_param.z;

    if (is_fixed == 1u) {
        return;
    }

    // let position = positions[index];
    let position = vec3<f32>(positions[index * 3u], positions[index * 3u + 1u], positions[index * 3u + 2u]);
    // var force = forces[index];
    var force = vec3<f32>(forces[index * 3u], forces[index * 3u + 1u], forces[index * 3u + 2u]);

    // dampening
    force *= params.dampening;

    // accumulate forces from springs
    for (var i = 0u; i < spring_count; i++) {
        let spring_id = spring_list[spring_pointer + i];
        // let spring_force = spring_forces[spring_id];
        let spring_force = vec3<f32>(
            spring_forces[spring_id * 3u],
            spring_forces[spring_id * 3u + 1u],
            spring_forces[spring_id * 3u + 2u]
        );
        let spring_vertex_ids = spring_vertex_ids[spring_id];

        let factor = select(-1.0, 1.0, spring_vertex_ids.x == index);
        force += spring_force * factor;
    }

    // gravity
    force.y -= 0.00005;

    // wind
    let noise_val = triNoise3D(position, 1.0, params.time) - 0.2;
    let wind_force = noise_val * 0.0001 * params.wind;
    force.z -= wind_force;

    // sphere interaction
    if (params.sphere_active > 0.0) {
        let sphere_pos = vec3<f32>(params.sphere_x, params.sphere_y, params.sphere_z);
        let new_pos = position + force;
        let delta_sphere = new_pos - sphere_pos;
        let dist = length(delta_sphere);

        if (dist < SPHERE_RADIUS) {
            let penetration = SPHERE_RADIUS - dist;
            let normal = delta_sphere / max(dist, 0.000001);
            force += normal * penetration;
        }
    }

    // forces[index] = force;
    forces[index * 3u] = force.x;
    forces[index * 3u + 1u] = force.y;
    forces[index * 3u + 2u] = force.z;

    // positions[index] += force;
    let new_position = position + force;
    positions[index * 3u] = new_position.x;
    positions[index * 3u + 1u] = new_position.y;
    positions[index * 3u + 2u] = new_position.z;
}
"""

# create the Verlet geometry and buffers
spring_shader = ComputeShader(spring_compute_wgsl, entry_point="compute_spring_forces")
vertex_shader = ComputeShader(vertex_compute_wgsl, entry_point="compute_vertex_forces")

spring_shader.set_resource(0, params_buffer)
spring_shader.set_resource(1, position_buffer)
spring_shader.set_resource(2, spring_vertex_ids_buffer)
spring_shader.set_resource(3, spring_rest_lengths_buffer)
spring_shader.set_resource(4, spring_force_buffer)

vertex_shader.set_resource(0, params_buffer)
vertex_shader.set_resource(1, position_buffer)
vertex_shader.set_resource(2, force_buffer)
vertex_shader.set_resource(3, vertex_params_buffer)
vertex_shader.set_resource(4, spring_list_buffer)
vertex_shader.set_resource(5, spring_vertex_ids_buffer)
vertex_shader.set_resource(6, spring_force_buffer)

vertex_shader.set_constant("SPHERE_RADIUS", sphere_radius)


def create_cloth_mesh_buffer():
    vertex_count = cloth_segments_x * cloth_segments_x

    verlet_vertex_ids = np.zeros((vertex_count, 4), dtype=np.uint32)
    indices = []

    def get_index(x, y):
        return y * cloth_segments_x + x

    for x in range(cloth_segments_x):
        for y in range(cloth_segments_y):
            index = get_index(x, y)
            verlet_vertex_ids[index, 0] = vertex_columns[x][y]["id"]
            verlet_vertex_ids[index, 1] = vertex_columns[x + 1][y]["id"]
            verlet_vertex_ids[index, 2] = vertex_columns[x][y + 1]["id"]
            verlet_vertex_ids[index, 3] = vertex_columns[x + 1][y + 1]["id"]

            if x > 0 and y > 0:
                # two triangles for each quad
                indices.append(
                    [get_index(x, y), get_index(x - 1, y), get_index(x - 1, y - 1)]
                )
                indices.append(
                    [get_index(x, y), get_index(x - 1, y - 1), get_index(x, y - 1)]
                )

    verlet_vertex_ids_buffer = gfx.Buffer(
        data=verlet_vertex_ids,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    cloth_position_buffer = gfx.Buffer(
        data=np.zeros((vertex_count, 3), dtype=np.float32),
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.VERTEX,
    )

    cloth_normal_buffer = gfx.Buffer(
        data=np.zeros((vertex_count, 3), dtype=np.float32),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    return (
        cloth_position_buffer,
        verlet_vertex_ids_buffer,
        cloth_normal_buffer,
        np.array(indices, dtype=np.uint32),
    )


(
    cloth_position_buffer,
    verlet_vertex_ids_buffer,
    cloth_normal_buffer,
    cloth_indices,
) = create_cloth_mesh_buffer()


cloth_compute_wgsl = """

@group(0) @binding(0) var<storage, read> vertex_positions: array<f32>;
@group(0) @binding(1) var<storage, read> vertex_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> normals: array<f32>;

@compute @workgroup_size(64)
fn compute_normals(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    if (index >= arrayLength(&vertex_ids) / 4u) {
        return;
    }

    let vertex_id = vec4<u32>(vertex_ids[index * 4u], vertex_ids[index * 4u + 1u], vertex_ids[index * 4u + 2u], vertex_ids[index * 4u + 3u]);

    let v0 = vec3f(vertex_positions[vertex_id.x * 3u], vertex_positions[vertex_id.x * 3u + 1u], vertex_positions[vertex_id.x * 3u + 2u]);
    let v1 = vec3f(vertex_positions[vertex_id.y * 3u], vertex_positions[vertex_id.y * 3u + 1u], vertex_positions[vertex_id.y * 3u + 2u]);
    let v2 = vec3f(vertex_positions[vertex_id.z * 3u], vertex_positions[vertex_id.z * 3u + 1u], vertex_positions[vertex_id.z * 3u + 2u]);
    let v3 = vec3f(vertex_positions[vertex_id.w * 3u], vertex_positions[vertex_id.w * 3u + 1u], vertex_positions[vertex_id.w * 3u + 2u]);

    let top = v0 + v1;
    let right = v1 + v3;
    let bottom = v2 + v3;
    let left = v0 + v2;

    let tangent = normalize(right - left);
    let bitangent = normalize(bottom - top);

    let normal = normalize(cross(tangent, bitangent));

    let position = (v0 + v1 + v2 + v3) / 4.0;

    // update normals
    normals[index * 3u] = normal.x;
    normals[index * 3u + 1u] = normal.y;
    normals[index * 3u + 2u] = normal.z;

    // update positions
    positions[index * 3u] = position.x;
    positions[index * 3u + 1u] = position.y;
    positions[index * 3u + 2u] = position.z;
}

"""

cloth_buffer_shader = ComputeShader(cloth_compute_wgsl, entry_point="compute_normals")

cloth_buffer_shader.set_resource(0, position_buffer)
cloth_buffer_shader.set_resource(1, verlet_vertex_ids_buffer)
cloth_buffer_shader.set_resource(2, cloth_position_buffer)
cloth_buffer_shader.set_resource(3, cloth_normal_buffer)


# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

cloth_geometry = gfx.Geometry(
    positions=cloth_position_buffer,
    indices=cloth_indices,
    normals=cloth_normal_buffer,
)

cloth_material = gfx.MeshPhysicalMaterial(
    # color="#030d37",
    color=(0.2, 0.4, 0.8),
    side="both",
    opacity=0.85,
    alpha_mode="blend",
    env_map=env_tex,
    env_map_intensity=3.0,
    sheen=1.0,
    sheen_roughness=0.5,
    sheen_color=(1.0, 1.0, 1.0),
    roughness=1.0,
)

cloth_mesh = gfx.Mesh(cloth_geometry, cloth_material)
scene.add(cloth_mesh)


def create_verlet_indices():
    indices = []

    for x in range(cloth_segments_x):
        for y in range(cloth_segments_y):
            v0 = x * (cloth_segments_y + 1) + y
            v1 = (x + 1) * (cloth_segments_y + 1) + y
            v2 = (x + 1) * (cloth_segments_y + 1) + y + 1
            v3 = x * (cloth_segments_y + 1) + y + 1

            indices.append([v0, v1, v2])
            indices.append([v0, v2, v3])

            indices.append([v0, v3, v1])
            indices.append([v1, v3, v2])

    return np.array(indices, dtype=np.uint32)


verlet_system_geometry = gfx.Geometry(
    positions=position_buffer,
    indices=create_verlet_indices(),
)
verlet_system_material = gfx.MeshBasicMaterial(
    side="both",
    wireframe=True,
)
verlet_system_point_material = gfx.PointsMarkerMaterial(
    size=8.0,
    # size_space="model",
    marker=gfx.MarkerShape.square,
)

verlet_system_mesh = gfx.Mesh(verlet_system_geometry, verlet_system_material)
verlet_system_points = gfx.Points(verlet_system_geometry, verlet_system_point_material)

verlet_system = gfx.Group()
verlet_system.add(verlet_system_mesh)
verlet_system.add(verlet_system_points)

verlet_system.visible = False  # Hide the Verlet system mesh by default
scene.add(verlet_system)


# create a sphere to interact with the cloth
sphere_geometry = gfx.sphere_geometry(radius=sphere_radius * 0.95)
sphere_material = gfx.MeshStandardMaterial(env_map=env_tex)
sphere = gfx.Mesh(sphere_geometry, sphere_material)
scene.add(sphere)

scene.add(gfx.Background.from_color((0.1, 0.1, 0.1, 1)))

camera.local.position = (-1.6, -0.1, -1.6)

controller = gfx.OrbitController(camera, register_events=renderer, target=(0, -0.1, 0))

gui_renderer = ImguiRenderer(renderer.device, canvas)

params_data = np.array([0.99, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 0.0], dtype=np.float32)
wireframe_mode = False
show_sphere = True

show_verlet_system = False


def draw_ui():
    global wireframe_mode, show_sphere, show_verlet_system

    imgui.set_next_window_size((350, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    imgui.begin("Cloth Controls")

    _, params_data[6] = imgui.slider_float("Stiffness", params_data[6], 0.1, 0.5)
    _, params_data[5] = imgui.slider_float("Wind", params_data[5], 0.0, 5.0)
    _, wireframe_mode = imgui.checkbox("Wireframe", wireframe_mode)

    verlet_system_changed, show_verlet_system = imgui.checkbox(
        "Show Verlet Mass-Spring System", show_verlet_system
    )
    if verlet_system_changed:
        verlet_system.visible = show_verlet_system
        cloth_mesh.visible = not show_verlet_system

    _, show_sphere = imgui.checkbox("Sphere", show_sphere)

    if imgui.collapsing_header("material", imgui.TreeNodeFlags_.default_open):
        _, cloth_material.color = imgui.color_edit3("Color", cloth_material.color.rgb)
        _, cloth_material.roughness = imgui.slider_float(
            "Roughness", cloth_material.roughness, 0.0, 1.0
        )
        _, cloth_material.sheen = imgui.slider_float(
            "Sheen", cloth_material.sheen, 0.0, 1.0
        )
        _, cloth_material.sheen_roughness = imgui.slider_float(
            "Sheen Roughness", cloth_material.sheen_roughness, 0.0, 1.0
        )
        _, cloth_material.sheen_color = imgui.color_edit3(
            "Sheen Color", cloth_material.sheen_color.rgb
        )

    imgui.end()


gui_renderer.set_gui(draw_ui)

clock = gfx.Clock()

timestamp = 0.0
time_since_last = 0.0

stat = Stats(align="right", device=renderer.device, canvas=canvas)


def animate():
    global wireframe_mode, show_sphere, timestamp, time_since_last

    with stat:
        dt = min(clock.get_delta(), 1 / 60)

        time_per_step = 1 / 360

        time_since_last += dt

        while time_since_last >= time_per_step:
            # run the simulation step
            timestamp += time_per_step
            time_since_last -= time_per_step

            params_data[7] = timestamp

            # update sphere position
            sphere_x = np.sin(timestamp * 2.1) * 0.1
            sphere_z = np.sin(timestamp * 0.8)
            sphere_y = 0.0

            sphere.local.position = (sphere_x, sphere_y, sphere_z)

            params_data[1] = sphere_x
            params_data[2] = sphere_y
            params_data[3] = sphere_z
            params_data[4] = 1.0 if show_sphere else 0.0

            # update parameters buffer
            params_buffer.set_data(params_data)

            # do the Verlet integration
            spring_shader.dispatch((spring_count + 63) // 64)
            vertex_shader.dispatch((vertex_count + 63) // 64)

        # update cloth mesh positions and normals
        cloth_buffer_shader.dispatch((cloth_segments_x * cloth_segments_x + 63) // 64)

        # update render state
        cloth_material.wireframe = wireframe_mode
        sphere.visible = show_sphere
        renderer.render(scene, camera)
        gui_renderer.render()

    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
