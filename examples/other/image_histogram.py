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
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.lineshader import LineShader


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

    def get_code(self):
        return (
            super()
            .get_code()
            .replace(
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
        )


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


def compute_histogram(img):
    start_time = time.time()
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    l = r * 0.299 + g * 0.587 + b * 0.114

    hist_r = np.histogram(r, bins=256, range=(0, 256))[0].astype(np.float32)
    hist_g = np.histogram(g, bins=256, range=(0, 256))[0].astype(np.float32)
    hist_b = np.histogram(b, bins=256, range=(0, 256))[0].astype(np.float32)
    hist_l = np.histogram(l, bins=256, range=(0, 256))[0].astype(np.float32)

    max_val = max(hist_r.max(), hist_g.max(), hist_b.max(), hist_l.max())
    hist_r = hist_r / max_val
    hist_g = hist_g / max_val
    hist_b = hist_b / max_val
    hist_l = hist_l / max_val

    computation_time = time.time() - start_time
    return hist_r, hist_g, hist_b, hist_l, computation_time


def update_histogram(hist_r, hist_g, hist_b, hist_l):
    positions = hist_line.geometry.positions.data.reshape(4, 257, 3)
    positions[0, :256, 1] = hist_r
    positions[1, :256, 1] = hist_g
    positions[2, :256, 1] = hist_b
    positions[3, :256, 1] = hist_l
    hist_line.geometry.positions.update_range()


hist_r, hist_g, hist_b, hist_l, computation_time = compute_histogram(img)
update_histogram(hist_r, hist_g, hist_b, hist_l)


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

            # Trigger recomputation of the histogram
            hist_r, hist_g, hist_b, hist_l, computation_time = compute_histogram(img)
            update_histogram(hist_r, hist_g, hist_b, hist_l)

        imgui.text(f"Histogram computation time: {computation_time * 1000:.1f} ms")

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
