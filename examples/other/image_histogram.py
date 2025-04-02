import wgpu
import ctypes

def compute_with_buffers(input_arrays, output_arrays, shader, n=None):
    """Apply the given compute shader to the given input_arrays and return
    output arrays. Both input and output arrays are represented on the GPU
    using storage buffer objects.

    Arguments:
        input_arrays (dict): A dict mapping int bindings to arrays. The array
            can be anything that supports the buffer protocol, including
            bytes, memoryviews, ctypes arrays and numpy arrays. The
            type and shape of the array does not need to match the type
            with which the shader will interpret the buffer data (though
            it probably makes your code easier to follow).
        output_arrays (dict): A dict mapping int bindings to output shapes.
            If the value is int, it represents the size (in bytes) of
            the buffer. If the value is a tuple, its last element
            specifies the format (see below), and the preceding elements
            specify the shape. These are used to ``cast()`` the
            memoryview object before it is returned. If the value is a
            ctypes array type, the result will be cast to that instead
            of a memoryview. Note that any buffer that is NOT in the
            output arrays dict will be considered readonly in the shader.
        shader (str or bytes): The shader as a string of WGSL code or SpirV bytes.
        n (int, tuple, optional): The dispatch counts. Can be an int
            or a 3-tuple of ints to specify (x, y, z). If not given or None,
            the length of the first output array type is used.

    Returns:
        output (dict): A dict mapping int bindings to memoryviews.

    The format characters to cast a ``memoryview`` are hard to remember, so
    here's a refresher:

    * "b" and "B" are signed and unsigned 8-bit ints.
    * "h" and "H" are signed and unsigned 16-bit ints.
    * "i" and "I" are signed and unsigned 32-bit ints.
    * "e" and "f" are 16-bit and 32-bit floats.
    """

    # Check input arrays
    if not isinstance(input_arrays, dict):  # empty is ok
        raise TypeError("input_arrays must be a dict.")
    for key, array in input_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of input_arrays must be int.")
        # Simply wrapping in a memoryview ensures that it supports the buffer protocol
        memoryview(array)

    # Check output arrays
    output_infos = {}
    if not isinstance(output_arrays, dict) or not output_arrays:
        raise TypeError("output_arrays must be a nonempty dict.")
    for key, array_descr in output_arrays.items():
        if not isinstance(key, int):
            raise TypeError("keys of output_arrays must be int.")
        if isinstance(array_descr, str) and "x" in array_descr:
            array_descr = tuple(array_descr.split("x"))
        if isinstance(array_descr, int):
            output_infos[key] = {
                "length": array_descr,
                "nbytes": array_descr,
                "format": "B",
                "shape": (array_descr,),
            }
        elif isinstance(array_descr, tuple):
            format = array_descr[-1]
            try:
                format_size = FORMAT_SIZES[format]
            except KeyError:
                raise ValueError(
                    f"Invalid format for output array {key}: {format}"
                ) from None
            shape = tuple(int(i) for i in array_descr[:-1])
            if not (shape and all(i > 0 for i in shape)):
                raise ValueError(f"Invalid shape for output array {key}: {shape}")
            nbytes = format_size
            for i in shape:
                nbytes *= i
            output_infos[key] = {
                "length": shape[0],
                "nbytes": nbytes,
                "format": format,
                "shape": shape,
            }
        elif isinstance(array_descr, type) and issubclass(array_descr, ctypes.Array):
            output_infos[key] = {
                "length": array_descr._length_,
                "nbytes": ctypes.sizeof(array_descr),
                "ctypes_array_type": array_descr,
            }
        else:
            raise TypeError(
                f"Invalid value for output array description: {array_descr}"
            )

    # Get nx, ny, nz from n
    if n is None:
        output_info = next(iter(output_infos.values()))
        nx, ny, nz = output_info["length"], 1, 1
    elif isinstance(n, int):
        nx, ny, nz = int(n), 1, 1
    elif isinstance(n, tuple) and len(n) == 3:
        nx, ny, nz = int(n[0]), int(n[1]), int(n[2])
    else:
        raise TypeError("compute_with_buffers: n must be None, an int, or 3-int tuple.")
    if not (nx >= 1 and ny >= 1 and nz >= 1):
        raise ValueError("compute_with_buffers: n value(s) must be >= 1.")

    # Create a device and compile the shader
    device = wgpu.utils.get_default_device()
    cshader = device.create_shader_module(code=shader)

    # Create buffers for input and output arrays
    buffers = {}
    for index, array in input_arrays.items():
        usage = wgpu.BufferUsage.STORAGE
        if index in output_arrays:
            usage |= wgpu.BufferUsage.COPY_SRC
        buffer = device.create_buffer_with_data(data=array, usage=usage)
        buffers[index] = buffer
    for index, info in output_infos.items():
        if index in input_arrays:
            continue  # We already have this buffer
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        buffers[index] = device.create_buffer(size=info["nbytes"], usage=usage)

    # Create bindings and binding layouts
    bindings = []
    binding_layouts = []
    for index, buffer in buffers.items():
        bindings.append(
            {
                "binding": index,
                "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
            }
        )
        storage_types = (
            wgpu.BufferBindingType.read_only_storage,
            wgpu.BufferBindingType.storage,
        )
        binding_layouts.append(
            {
                "binding": index,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": storage_types[index in output_infos],
                    "has_dynamic_offset": False,
                },
            }
        )

    # Put buffers together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create a pipeline and "run it"
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(nx, ny, nz)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])

    # Read the current data of the output buffers
    output = {}
    for index, info in output_infos.items():
        buffer = buffers[index]
        # m = buffer.read_data()  # old API
        m = device.queue.read_buffer(buffer)  # slow, can also be done async
        if "ctypes_array_type" in info:
            output[index] = info["ctypes_array_type"].from_buffer(m)
        else:
            output[index] = m.cast(info["format"], shape=info["shape"])

    return output


FORMAT_SIZES = {"b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4, "e": 2, "f": 4}

# It's tempting to allow for other formats, like "int32" and "f4", but
# users who like numpy will simply specify the number of bytes and
# convert the result. Users who will work with the memoryview directly
# should not be confused with other formats than memoryview.cast()
# normally supports.

"""
Image Histogram Analysis
=======================

The goal is to move the histogram computation to the GPU.
In the given example, it is all done on the CPU.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import time
import numpy as np
import imageio.v3 as imageio
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui

import pygfx as gfx
from pylinalg import vec_transform, vec_unproject
from pygfx.renderers.wgpu import register_wgpu_render_function, Binding
from pygfx.renderers.wgpu.shaders.lineshader import LineShader
from pygfx.resources import Buffer
from pygfx.utils import array_from_shadertype

renderer_uniform_type = dict(last_i="i4")


from wgpu.utils.compute import compute_with_buffers

# Compute shader for histogram computation
histogram_shader = """
@group(0) @binding(0)
var<storage, read> input_image: array<f32>;
@group(0) @binding(1)
var<storage, read_write> histogram_r: array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> histogram_g: array<atomic<u32>>;
@group(0) @binding(3)
var<storage, read_write> histogram_b: array<atomic<u32>>;
@group(0) @binding(4)
var<storage, read_write> histogram_l: array<atomic<u32>>;
@group(0) @binding(5)
var<storage, read> image_shape: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = image_shape[0];
    let height = image_shape[1];
    let total_pixels = width * height;
    
    // Each workgroup processes 256 pixels
    let start_idx = gid.x * 256;
    let end_idx = min(start_idx + 256, total_pixels);
    
    for (var i = start_idx; i < end_idx; i++) {
        let r = input_image[i * 3];
        let g = input_image[i * 3 + 1];
        let b = input_image[i * 3 + 2];
        let l = 0.299 * r + 0.587 * g + 0.114 * b;
        
        // Convert to bin index (0-255)
        let r_bin = u32(r);
        let g_bin = u32(g);
        let b_bin = u32(b);
        let l_bin = u32(l);
        
        // Atomic increment for each histogram
        atomicAdd(&histogram_r[r_bin], 1u);
        atomicAdd(&histogram_g[g_bin], 1u);
        atomicAdd(&histogram_b[b_bin], 1u);
        atomicAdd(&histogram_l[l_bin], 1u);
    }
}
"""

def compute_histogram_gpu(img):
    """Compute histogram using GPU compute shader."""
    start_time = time.time()
    if img.shape[-1] != 3:
        img = np.repeat(img, 3, axis=-1)
    
    # Reshape image to 1D array of RGB values
    img_flat = img.reshape(-1, 3).astype(np.float32)
    
    # Create bindings for compute shader
    bindings = {
        0: img_flat,  # input image
        1: np.zeros(256, dtype=np.uint32),  # histogram_r
        2: np.zeros(256, dtype=np.uint32),  # histogram_g
        3: np.zeros(256, dtype=np.uint32),  # histogram_b
        4: np.zeros(256, dtype=np.uint32),  # histogram_l
        5: np.array([img.shape[1], img.shape[0]], dtype=np.uint32),  # image shape
    }
    
    # Run compute shader
    out = compute_with_buffers(
        input_arrays=bindings,
        output_arrays={1: (256, "f"), 2: (256, "f"), 3: (256, "f"), 4: (256, "f")},
        shader=histogram_shader,
        n=(img_flat.shape[0] // 256 + 1, 1, 1),
    )
    
    # Get histograms from output
    hist_r = np.frombuffer(out[1], dtype=np.float32)
    hist_g = np.frombuffer(out[2], dtype=np.float32)
    hist_b = np.frombuffer(out[3], dtype=np.float32)
    hist_l = np.frombuffer(out[4], dtype=np.float32)
    
    # Normalize histograms
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), hist_l.max())
    hist_r /= max_val
    hist_g /= max_val
    hist_b /= max_val
    hist_l /= max_val
    
    computation_time = time.time() - start_time
    return hist_r, hist_g, hist_b, hist_l, computation_time

# Get list of available standard images
standard_images = [
    "astronaut",
    "camera",
    "checkerboard",
    "clock",
    "coffee",
    "horse",
    "hubble_deep_field",
    "immunohistochemistry",
    "moon",
    "page",
    "text",
]

# Initialize canvas and renderer
canvas = WgpuCanvas(size=(800, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Create viewports for image and histogram
w, h = canvas.get_logical_size()
viewport_image = gfx.Viewport(renderer, rect=(0, 0, w // 2, h))
viewport_hist = gfx.Viewport(renderer, rect=(w // 2, 0, w - w // 2, h))

# Create scenes
scene_image = gfx.Scene()
scene_hist = gfx.Scene()

# Add background to scenes
scene_image.add(gfx.Background.from_color("#111111"))
scene_hist.add(gfx.Background.from_color("#111111"))

# Create camera for image view
camera_image = gfx.OrthographicCamera(w // 2, h)

# Create camera for histogram view
camera_hist = gfx.OrthographicCamera(256, 256)

# Create controllers
controller_image = gfx.PanZoomController(camera_image, register_events=viewport_image)
controller_hist = gfx.PanZoomController(camera_hist, register_events=viewport_hist)

# Create grid and rulers for histogram view
grid = gfx.Grid(
    None,
    gfx.GridMaterial(
        major_step=50,  # Major grid lines every 50 units
        minor_step=10,  # Minor grid lines every 10 units
        thickness_space="screen",
        major_thickness=2,
        minor_thickness=0.5,
        infinite=True,
    ),
    orientation="xy",
)
grid.local.z = -1

rulerx = gfx.Ruler(tick_side="right")
rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)

scene_hist.add(grid, rulerx, rulery)


def load_image(image_name):
    return imageio.imread(f"imageio:{image_name}.png")


# Create initial image and histogram
current_image_name = standard_images[0]
img = load_image(current_image_name)
image_object = gfx.Image(
    gfx.Geometry(grid=gfx.Texture(img, dim=2)), gfx.ImageBasicMaterial(clim=(0, 255))
)
image_object.local.scale_y = -1
scene_image.add(image_object)

# Update camera to show the full image
camera_image.show_object(image_object)

# Create histogram objects
x = np.arange(257, dtype=np.float32)
y = np.zeros_like(x)
z = np.zeros_like(x)

histogram_data = np.vstack(
    (
        np.column_stack((x, y, z)),  # red
        np.column_stack((x, y, z)),  # green
        np.column_stack((x, y, z)),  # blue
        np.column_stack((x, y, z)),  # luminance
    )
)

histogram_data[256::257, :] = np.nan


class HistogramMaterial(gfx.LineMaterial):
    uniform_type = dict(
        gfx.LineMaterial.uniform_type,
        absolute_scale="f4",
    )

    def __init__(self, *args, absolute_scale=1.0, log_scale=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_scale = log_scale
        self.absolute_scale = absolute_scale

    @property
    def log_scale(self):
        return self._store.log_scale

    @log_scale.setter
    def log_scale(self, value):
        self._store.log_scale = value

    @property
    def absolute_scale(self):
        return float(self.uniform_buffer.data["absolute_scale"])

    @absolute_scale.setter
    def absolute_scale(self, value):
        self.uniform_buffer.data["absolute_scale"] = float(value)
        self.uniform_buffer.update_full()


class Histogram(gfx.Line):
    pass


@register_wgpu_render_function(Histogram, HistogramMaterial)
class HistogramShader(LineShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        material = wobject.material

        self["log_scale"] = material.log_scale
        self["absolute_scale"] = material.absolute_scale

        self["instanced"] = False

    def get_bindings(self, wobject, shared):
        material = wobject.material
        geometry = wobject.geometry

        positions1 = geometry.positions

        # With vertex buffers, if a shader input is vec4, and the vbo has
        # Nx2, the z and w element will be zero. This works, because for
        # vertex buffers we provide additional information about the
        # striding of the data.
        # With storage buffers (aka SSBO) we just have some bytes that we
        # read from/write to in the shader. This is more free, but it means
        # that the data in the buffer must match with what the shader
        # expects. In addition to that, there's this thing with vec3's which
        # are padded to 16 bytes. So we either have to require our users
        # to provide Nx4 data, or read them as an array of f32.
        # Anyway, extra check here to make sure the data matches!
        if positions1.data.shape[1] != 3:
            raise ValueError(
                "For rendering (thick) lines, the geometry.positions must be Nx3."
            )

        uniform_buffer = Buffer(
            array_from_shadertype(renderer_uniform_type), force_contiguous=True
        )
        uniform_buffer.data["last_i"] = positions1.nitems - 1

        rbuffer = "buffer/read_only_storage"
        bindings = [
            Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            Binding("u_material", "buffer/uniform", material.uniform_buffer),
            Binding("u_renderer", "buffer/uniform", uniform_buffer),
            Binding("s_positions", rbuffer, positions1, "VERTEX"),
        ]

        # Per-vertex color, colormap, or a uniform color?
        if self["color_mode"] in ("vertex", "face"):
            bindings.append(Binding("s_colors", rbuffer, geometry.colors, "VERTEX"))
        elif self["color_mode"] in ("vertex_map", "face_map"):
            bindings.append(
                Binding("s_texcoords", rbuffer, geometry.texcoords, "VERTEX")
            )
            bindings.extend(
                self.define_generic_colormap(material.map, geometry.texcoords)
            )

        # Need a buffer for the loop and/or cumdist?
        if hasattr(self, "line_loop_buffer"):
            bindings.append(Binding("s_loop", rbuffer, self.line_loop_buffer, "VERTEX"))
        if hasattr(self, "line_distance_buffer"):
            bindings.append(
                Binding("s_cumdist", rbuffer, self.line_distance_buffer, "VERTEX")
            )

        bindings = {i: b for i, b in enumerate(bindings)}
        self.define_bindings(0, bindings)

        # Instanced lines have an extra storage buffer that we add manually
        bindings1 = {}  # non-auto-generated bindings
        if self["instanced"]:
            bindings1[0] = Binding(
                "s_instance_infos", rbuffer, wobject.instance_buffer, "VERTEX"
            )
        print('bindings', bindings)
        print('bindings1', bindings1)
        return {
            0: bindings,
            1: bindings1,
        }

    def get_code(self):
        code = super().get_code()
        # Pygfx 0.10.0 (not yet released)
        code = code.replace(
                """
    let pos_m_prev = load_s_positions(node_index_prev);
    let pos_m_node = load_s_positions(node_index);
    let pos_m_next = load_s_positions(node_index_next);
""",
                """
    let pos_m_prev_raw = load_s_positions(node_index_prev);
    let pos_m_node_raw = load_s_positions(node_index);
    let pos_m_next_raw = load_s_positions(node_index_next);

    let pos_m_prev = vec3<f32>(
        pos_m_prev_raw.x, 
    $$ if log_scale
        log(pos_m_prev_raw.y + 1.0) * u_material.absolute_scale,
    $$ else
        pos_m_prev_raw.y * u_material.absolute_scale,
    $$ endif
        pos_m_prev_raw.z,
    );
    let pos_m_node = vec3<f32>(
        pos_m_node_raw.x, 
    $$ if log_scale
        log(pos_m_node_raw.y + 1.0) * u_material.absolute_scale,
    $$ else
        pos_m_node_raw.y * u_material.absolute_scale,
    $$ endif
        pos_m_node_raw.z,
    );
    let pos_m_next = vec3<f32>(
        pos_m_next_raw.x, 
    $$ if log_scale
        log(pos_m_next_raw.y + 1.0) * u_material.absolute_scale,
    $$ else
        pos_m_next_raw.y * u_material.absolute_scale,
    $$ endif
        pos_m_next_raw.z,
    );
""",
        )
        # pygfx 0.9.0 -- stable version
        code = code.replace(
            """
    let pos_m_prev = load_s_positions(max(0, node_index - 1));
    let pos_m_node = load_s_positions(node_index);
    let pos_m_next = load_s_positions(min(u_renderer.last_i, node_index + 1));
""", """
    let pos_m_prev_raw = load_s_positions(max(0, node_index - 1));
    let pos_m_node_raw = load_s_positions(node_index);
    let pos_m_next_raw = load_s_positions(min(u_renderer.last_i, node_index + 1));

    let pos_m_prev = vec3<f32>(
        pos_m_prev_raw.x, 
    $$ if log_scale
        log(pos_m_prev_raw.y + 1.0) * u_material.absolute_scale,
    $$ else
        pos_m_prev_raw.y * u_material.absolute_scale,
    $$ endif
        pos_m_prev_raw.z,
    );
    let pos_m_node = vec3<f32>(
        pos_m_node_raw.x, 
        $$ if log_scale
        log(pos_m_node_raw.y + 1.0) * u_material.absolute_scale,
        $$ else
        pos_m_node_raw.y * u_material.absolute_scale,
        $$ endif
        pos_m_node_raw.z,
    );
    let pos_m_next = vec3<f32>(
        pos_m_next_raw.x, 
        $$ if log_scale
        log(pos_m_next_raw.y + 1.0) * u_material.absolute_scale,
        $$ else
        pos_m_next_raw.y * u_material.absolute_scale,
        $$ endif
        pos_m_next_raw.z,
    );
""",
        )
        return code


vertex_color = np.zeros((4, 257, 3), dtype=np.float32)
vertex_color[0, :256, 0] = 1
vertex_color[1, :256, 1] = 1
vertex_color[2, :256, 2] = 1
vertex_color[3, :256, :] = 1

hist_line = Histogram(
    gfx.Geometry(positions=histogram_data, colors=vertex_color.reshape(-1, 3)),
    HistogramMaterial(color=(1, 1, 1), color_mode="vertex", absolute_scale=255),
)
scene_hist.add(hist_line)

# State variables
use_log_scale = False
current_image_index = 0


def update_histogram(hist_r, hist_g, hist_b, hist_l):
    positions = hist_line.geometry.positions.data.reshape(4, 257, 3)
    positions[0, :256, 1] = hist_r
    positions[1, :256, 1] = hist_g
    positions[2, :256, 1] = hist_b
    positions[3, :256, 1] = hist_l
    hist_line.geometry.positions.update_range()

computation_time = 0

def draw_imgui():
    global current_image_index
    global img, hist_r, hist_g, hist_b, hist_l, computation_time
    global use_log_scale

    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)

    if is_expand:
        # Image selection dropdown
        changed, current_image_index = imgui.combo(
            "Image", current_image_index, standard_images, len(standard_images)
        )
        # Log scale toggle
        log_changed, use_log_scale = imgui.checkbox("Log Scale", use_log_scale)
        if log_changed:
            hist_line.material.log_scale = use_log_scale

        if changed:
            img = load_image(standard_images[current_image_index])
            image_object.geometry.grid = gfx.Texture(img, dim=2)

            # Trigger recomputation of the histogram using GPU
            hist_r, hist_g, hist_b, hist_l, computation_time = compute_histogram_gpu(img)
            update_histogram(hist_r, hist_g, hist_b, hist_l)

        imgui.text(f"GPU Histogram computation time: {computation_time * 1000:.1f} ms")

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


def map_screen_to_world(pos, viewport_size):
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)

    pos_ndc += vec_transform(camera_hist.world.position, camera_hist.camera_matrix)
    pos_world = vec_unproject(pos_ndc[:2], camera_hist.camera_matrix)

    return pos_world


# Create GUI renderer
gui_renderer = ImguiRenderer(renderer.device, canvas)

# Compute histogram at startup to have correct values
hist_r, hist_g, hist_b, hist_l, computation_time = compute_histogram_gpu(img)
update_histogram(hist_r, hist_g, hist_b, hist_l)


def animate():
    w, h = canvas.get_logical_size()

    viewport_image.rect = (0, 0, w // 2, h)
    viewport_hist.rect = (w // 2, 0, w - w // 2, h)

    # Update rulers and grid for histogram view
    xmin, ymin = 0, h
    xmax, ymax = w // 2, 0

    world_xmin, world_ymin, _ = map_screen_to_world((xmin, ymin), (w // 2, h))
    world_xmax, world_ymax, _ = map_screen_to_world((xmax, ymax), (w // 2, h))

    # Set start and end positions of rulers
    rulerx.start_pos = world_xmin, 0, -1
    rulerx.end_pos = world_xmax, 0, -1
    rulerx.start_value = rulerx.start_pos[0]
    statsx = rulerx.update(camera_hist, (w // 2, h))

    rulery.start_pos = 0, world_ymin, -1
    rulery.end_pos = 0, world_ymax, -1
    rulery.start_value = rulery.start_pos[1]
    statsy = rulery.update(camera_hist, (w // 2, h))

    # Update grid steps based on ruler stats
    major_step_x, major_step_y = statsx["tick_step"], statsy["tick_step"]
    grid.material.major_step = major_step_x, major_step_y
    grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y

    viewport_image.render(scene_image, camera_image)
    viewport_hist.render(scene_hist, camera_hist)

    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


gui_renderer.set_gui(draw_imgui)

if __name__ == "__main__":
    canvas.request_draw(animate)
    run()
