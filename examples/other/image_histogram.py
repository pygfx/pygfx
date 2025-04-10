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


# Create initial image texture.
# Note how we set the usage. We set STORAGE_BINDING so we can use it in a compute  shader.
# We also set TEXTURE_BINDING because we also want to rendet the image.
image_texture = gfx.Texture(
    load_image(standard_images[0]),
    dim=2,
    usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
)

# Create image object from the texture.
# The image object stays the same, we swap out its texture when the user selects an image.
image_object = gfx.Image(
    gfx.Geometry(grid=image_texture),
    gfx.ImageBasicMaterial(clim=(0, 255)),
)
image_object.local.scale_y = -1
scene.add(image_object)


# Create a buffer to store the line representing the histogram
nbins = 20
histogram_buffer = gfx.Buffer(
    nbytes=nbins * 3 * 4, nitems=nbins, format="3xf4", usage=wgpu.BufferUsage.STORAGE
)

# Create the line object with that buffer
histogram_object = gfx.Line(
    gfx.Geometry(positions=histogram_buffer),
    gfx.LineMaterial(color="yellow"),
)
histogram_object.local.y += 10
histogram_object.local.scale_y = 50
histogram_object.local.scale_x = image_texture.size[0] / (nbins - 1)
scene.add(histogram_object)


# Update camera to show the image and histogram
camera.show_object(scene)


## GPU Compute

histogram_wgsl = """

@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> s_positions: array<f32>;

// from: https://www.w3.org/WAI/GL/wiki/Relative_luminance
const kSRGBLuminanceFactors = vec3f(0.2126, 0.7152, 0.0722);
fn srgbLuminance(color: vec3f) -> f32 {
    return saturate(dot(color, kSRGBLuminanceFactors));
}

override bin_count: u32 = 20u;
override flip_xy: bool = false;

@compute @workgroup_size(1)
fn main() {
    let size = textureDimensions(imageTexture, 0);
    //let hist_size = textureDimensions(histTexture, 0);
    let numBins = i32(bin_count);  //hist_size.x;
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

    // Fill output positions buffer
    for (var i = 0; i < 20; i++) {
        let count = bins[i];
        let x = f32(i);
        let y = f32(count) / f32(maxCount);
        s_positions[i*3] = select(x, y, flip_xy);
        s_positions[i*3+1] = select(y, x, flip_xy);
    }
}
"""


from typing import Optional, Union


# TODO: ability to concatenate multiple steps
# TODO: not sure about the name.
# TODO: move this into Pygfx
class ComputeStep:
    """Abstraction for a compute shader.

    Parameters
    ----------
    wgsl : str
        The compute shader's code as WGSL.
    entry_point : str | None
        The name of the wgsl function that must be called.
        If the wgsl code has only one entry-point (a function marked with ``@compute``)
        this argument can be omitted.
    """

    def __init__(self, wgsl, *, entry_point: Optional[str] = None):
        # Fixed
        self._wgsl = wgsl
        self._entry_point = entry_point

        # Things that can be changed via the API
        self._resources = {}
        self._constants = {}

        # Flag to keep track whether this object changed.
        # Note that this says nothing about the contents of buffers/textures used as input.
        self._changed = True

        # Internal variables
        self._device = None
        self._shader_module = None
        self._pipeline = None
        self._bind_group = None

    @property
    def changed(self) -> bool:
        return self._changed

    def set_resource(
        self,
        index: int,
        resource: Union[gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture],
    ):
        # Check
        if not isinstance(index, int):
            raise TypeError(f"ComputeStep resource index must be int, not {index!r}.")
        if not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture)
        ):
            raise TypeError(
                f"ComputeStep resource value must be gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, or wgpu.GPUTexture, not {resource!r}"
            )

        # Update if different
        old_resource = self._resources.get(index)
        if resource is not old_resource:
            if resource is None:
                self._resources.pop(index, None)
            else:
                self._resources[index] = resource
            self._bind_group = None
            self._changed = True

    def set_constant(self, name: str, value: Union[bool, int, float, None]):
        # NOTE: we could also provide support for uniform variables.
        # The override constants are nice and simple, but require the pipeline
        # to be re-created whenever a contant changes.

        # Check
        if not isinstance(name, str):
            raise TypeError(f"ComputeStep constant name must be str, not {name!r}.")
        if not (value is None or isinstance(value, (bool, int, float))):
            raise TypeError(
                f"ComputeStep constant value must be bool, int, float, or None, not {value!r}."
            )

        # Update if different
        old_value = self._constants.get(name)
        if value != old_value:
            if value is None:
                self._constants.pop(name, None)
            else:
                self._constants[name] = value
            self._pipeline = None
            self._changed = True

    def _get_bindings_from_resources(self):
        bindings = []
        for index, resource in self._resources.items():
            # Get native buffer or texture
            if isinstance(resource, gfx.Resource):
                wgpu_object = gfx.renderers.wgpu.engine.update.ensure_wgpu_object(
                    resource
                )
            else:
                wgpu_object = resource  # wgpu.GPUBuffer or wgpu.GPUTexture

            # todo: maybe check usage here so we can provide more useful message?
            if isinstance(wgpu_object, wgpu.GPUBuffer):
                bindings.append(
                    {
                        "binding": index,
                        "resource": {
                            "buffer": wgpu_object,
                            "offset": 0,
                            "size": wgpu_object.size,
                        },
                    }
                )
            elif isinstance(wgpu_object, wgpu.GPUTexture):
                bindings.append(
                    {
                        "binding": index,
                        "resource": wgpu_object.create_view(
                            usage=wgpu.TextureUsage.STORAGE_BINDING
                        ),
                    }
                )
            else:
                raise RuntimeError(f"Unexpected resource: {resource}")
        return bindings

    def dispatch(self, nx, ny=1, nz=1):
        nx, ny, nz = int(nx), int(ny), int(nz)

        # Reset
        self._changed = False

        # Get device
        if self._device is None:
            self._shader_module = None
            self._device = gfx.renderers.wgpu.Shared.get_instance().device
        device = self._device

        # Compile the shader
        if self._shader_module is None:
            self._pipeline = None
            self._shader_module = device.create_shader_module(code=self._wgsl)

        # Get the pipeline object
        if self._pipeline is None:
            self._bind_group = None
            self._pipeline = device.create_compute_pipeline(
                layout="auto",
                compute={
                    "module": self._shader_module,
                    "entry_point": self._entry_point,
                    "constants": self._constants,
                },
            )

        # Get the bind group object
        if self._bind_group is None:
            bind_group_layout = self._pipeline.get_bind_group_layout(0)
            bindings = self._get_bindings_from_resources()
            self._bind_group = device.create_bind_group(
                layout=bind_group_layout, entries=bindings
            )

        # Make sure that all used resources have a wgpu-representation, and are synced
        for resource in self._resources.values():
            gfx.renderers.wgpu.engine.update.update_resource(resource)

        # Run ...
        command_encoder = device.create_command_encoder()
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, self._bind_group)
        compute_pass.dispatch_workgroups(nx, ny, nz)
        compute_pass.end()
        device.queue.submit([command_encoder.finish()])


histogram_compute = ComputeStep(histogram_wgsl)
histogram_compute.set_resource(0, image_object.geometry.grid)
histogram_compute.set_resource(1, histogram_buffer)


## ------

current_image_index = 0


def draw_imgui():
    global current_image_index

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
            # Create new texture object
            img = load_image(standard_images[current_image_index])
            image_texture = gfx.Texture(
                img,
                dim=2,
                usage=wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            # Update the image
            image_object.geometry.grid = image_texture
            # Update histogram
            histogram_compute.set_resource(0, image_object.geometry.grid)
            histogram_object.local.scale_x = image_texture.size[0] / (nbins - 1)

        # imgui.text(f"Histogram computation time: {computation_time * 1000:.1f} ms")

    imgui.end()
    imgui.end_frame()
    imgui.render()
    return imgui.get_draw_data()


# Create GUI renderer
gui_renderer = ImguiRenderer(renderer.device, canvas)
gui_renderer.set_gui(draw_imgui)


def animate():
    if histogram_compute.changed:
        histogram_compute.dispatch(1)

    renderer.render(scene, camera)
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
