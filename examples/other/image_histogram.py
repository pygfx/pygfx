"""
Image Histogram Analysis
=======================

The goal is to move the histogram computation to the GPU.
In the given example, it is all done on the CPU.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
import imageio.v3 as imageio
from rendercanvas.auto import RenderCanvas, loop
import wgpu
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui

import pygfx as gfx


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
canvas = RenderCanvas(size=(800, 600))
renderer = gfx.renderers.WgpuRenderer(canvas)

# Scene, camera, controller
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#777"))
camera = gfx.OrthographicCamera()
controller = gfx.PanZoomController(camera, register_events=renderer)


def load_image(image_name):
    im = imageio.imread(f"imageio:{image_name}.png")
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)
    return im


# Create initial image
img = load_image(standard_images[0])
image_texture = gfx.Texture(img, dim=2, usage=wgpu.TextureUsage.STORAGE_BINDING)
image_object = gfx.Image(
    gfx.Geometry(grid=image_texture),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
image_object.local.scale_y = -1
scene.add(image_object)


# Create an image object to contain the histogram
nbins = 20
histogram_texture = gfx.Texture(
    data=None,
    dim=2,
    size=(nbins, 100, 1),
    format="rgba8unorm",
    usage=wgpu.TextureUsage.STORAGE_BINDING,
)
histogram_object = gfx.Image(
    gfx.Geometry(grid=histogram_texture),
    gfx.ImageBasicMaterial(clim=(0, 255), interpolation="nearest"),
)
histogram_object.local.y += 10
histogram_object.local.scale_x = 512 / nbins
scene.add(histogram_object)


# Update camera to show the image and histogram
camera.show_object(scene)


## GPU Compute

histogram_wgsl = """

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var histTexture: texture_storage_2d<rgba8unorm,write>;

// from: https://www.w3.org/WAI/GL/wiki/Relative_luminance
const kSRGBLuminanceFactors = vec3f(0.2126, 0.7152, 0.0722);
fn srgbLuminance(color: vec3f) -> f32 {
    return saturate(dot(color, kSRGBLuminanceFactors));
}

@compute @workgroup_size(1)
fn main() {
    let size = textureDimensions(imageTexture, 0);
    //let hist_size = textureDimensions(histTexture, 0);
    let numBins = 20;//hist_size.x;
    let numLevels = 100;//hist_size.y;

    // Compute hist

    var bins: array<i32,20>;

    let lastBinIndex = u32(numBins - 1);
    for (var y = 0u; y < size.y; y++) {
        for (var x = 0u; x < size.x; x++) {
        let position = vec2u(x, y);
        let color = textureLoad(imageTexture, position, 0);
        let v = srgbLuminance(color.rgb);
        let bin = min(u32(v * f32(numBins)), lastBinIndex);
        bins[bin] += 1;
        }
    }

    // Draw hist

    // Detect maximum count
    var maxCount = 0;
    for (var x = 0; x < numBins; x++) {
        maxCount = max(maxCount, bins[x]);
    }

    // Fill output image
    for (var x = 0; x < numBins; x++) {
        let count = bins[x];
        let max_y = i32(round( f32(numLevels) * f32(count) / f32(maxCount) ));
        for (var y = 0; y < max_y; y++) {
            let value = vec4<f32>(1.0, 1.0, 0.0, 1.0);
            textureStore(histTexture, vec2<i32>(x, y), value);
        }
        for (var y = max_y; y < numLevels; y++) {
            let value = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            textureStore(histTexture, vec2<i32>(x, y), value);
        }

    }
}

"""


def compute_histogram():
    nx, ny, nz = 100, 1, 1

    # Get Pygfx's wgu device object
    device = gfx.renderers.wgpu.Shared.get_instance().device

    # TODO: accessing internal _wgpu_object attribute
    image_wgpu_texture = image_object.geometry.grid._wgpu_object
    histogram_wgpu_texture = histogram_texture._wgpu_object

    # Compile our shader
    # TODO: don't do this every time!
    cshader = device.create_shader_module(code=histogram_wgsl)
    compute = {
        "module": cshader,
        "entry_point": "main",
    }
    # compute["constants"] = constants

    # Create a pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout="auto",
        compute=compute,
    )

    # Create bindings
    bindings = [
        {
            "binding": 0,
            "resource": image_wgpu_texture.create_view(
                usage=wgpu.TextureUsage.STORAGE_BINDING
            ),
        },
        {
            "binding": 1,
            "resource": histogram_wgpu_texture.create_view(
                usage=wgpu.TextureUsage.STORAGE_BINDING
            ),
        },
    ]
    bind_group_layout = compute_pipeline.get_bind_group_layout(0)
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Run
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(nx, ny, nz)
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])


## ------

current_image_index = 0
hist_needs_update = True


def draw_imgui():
    global current_image_index
    global hist_needs_update

    imgui.new_frame()

    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos((0, 0), imgui.Cond_.always)

    is_expand, _ = imgui.begin("Controls", None)

    if is_expand:
        # Image selection dropdown
        changed, current_image_index = imgui.combo(
            "Image", current_image_index, standard_images, len(standard_images)
        )
        if changed:
            img = load_image(standard_images[current_image_index])
            image_texture = gfx.Texture(
                img, dim=2, usage=wgpu.TextureUsage.STORAGE_BINDING
            )
            image_object.geometry.grid = image_texture

            # Trigger recomputation of the histogram
            hist_needs_update = True

        # imgui.text(f"Histogram computation time: {computation_time * 1000:.1f} ms")

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


# Create GUI renderer
gui_renderer = ImguiRenderer(renderer.device, canvas)
gui_renderer.set_gui(draw_imgui)


def animate():
    global hist_needs_update

    renderer.render(scene, camera)

    if hist_needs_update:
        hist_needs_update = False
        compute_histogram()

    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
