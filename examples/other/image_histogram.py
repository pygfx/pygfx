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
from rendercanvas.auto import RenderCanvas, loop
import wgpu
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui

import pygfx as gfx


## Compute API to be moved elsewhere

from typing import Optional, Union


# TODO: move this into wgpu/pygfx/new lib
# TODO: not sure about the name.
# TODO: ability to concatenate multiple steps
# TODO: add support for uniforms
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
    label : str | None
        The label for this shader. Used to set labels of underlying wgpu objects,
        and in debugging messages.
    report_time : bool
        When set to True, will print the spent time to run the shader.
    """

    def __init__(
        self,
        wgsl,
        *,
        entry_point: Optional[str] = None,
        label: str = Optional[None],
        report_time: bool = False,
    ):
        # Fixed
        self._wgsl = wgsl
        self._entry_point = entry_point
        self._label = label
        self._report_time = report_time

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
        """Whether the shader has been changed.

        This can be a new value for a constant, or a different resource.
        Note that this says nothing about the values inside a buffer or texture resource.
        This value is reset when ``dispatch()`` is called.
        """
        return self._changed

    def set_resource(
        self,
        index: int,
        resource: Union[gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture],
        *,
        clear=False,
    ):
        """Set a resource.

        Parameters
        ----------
        index : int
            The binding index to connect this resource to. (The group is hardcoded to zero for now.)
        resource : buffer | texture
            The buffer or texture to attach. Can be a wgpu or pygfx resource.
        clear : bool
            When set to True (only possible for a buffer), the resource is cleared to zeros
            right before running the shader.
        """
        # Check
        if not isinstance(index, int):
            raise TypeError(f"ComputeStep resource index must be int, not {index!r}.")
        if not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture)
        ):
            raise TypeError(
                f"ComputeStep resource value must be gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, or wgpu.GPUTexture, not {resource!r}"
            )
        clear = bool(clear)
        if clear and not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer)
        ):
            raise ValueError("Can only clear a buffer, not a texture.")

        # Value to store
        new_value = resource, bool(clear)

        # Update if different
        old_value = self._resources.get(index)
        if new_value != old_value:
            if resource is None:
                self._resources.pop(index, None)
            else:
                self._resources[index] = new_value
            self._bind_group = None
            self._changed = True

    def set_constant(self, name: str, value: Union[bool, int, float, None]):
        """Set override constant.

        Setting override constants don't require shader recompilation, but does
        require re-creating the pipeline object. So it's less suited for things
        that change on every draw.
        """
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

    def _get_native_resource(self, resource):
        if isinstance(resource, gfx.Resource):
            return gfx.renderers.wgpu.engine.update.ensure_wgpu_object(resource)
        return resource

    def _get_bindings_from_resources(self):
        bindings = []
        for index, (resource, _) in self._resources.items():
            # Get wgpu.GPUBuffer or wgpu.GPUTexture
            wgpu_object = self._get_native_resource(resource)
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
        """Dispatch the workgroups, i.e. run the shader."""
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
            self._shader_module = device.create_shader_module(
                label=self._label, code=self._wgsl
            )

        # Get the pipeline object
        if self._pipeline is None:
            self._bind_group = None
            self._pipeline = device.create_compute_pipeline(
                label=self._label,
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
                label=self._label, layout=bind_group_layout, entries=bindings
            )

        # Make sure that all used resources have a wgpu-representation, and are synced
        for resource, _ in self._resources.values():
            if isinstance(resource, gfx.Resource):
                gfx.renderers.wgpu.engine.update.update_resource(resource)

        t0 = time.perf_counter()

        # Start!
        command_encoder = device.create_command_encoder(label=self._label)

        # Maybe clear some buffers
        for resource, clear in self._resources.values():
            if clear:
                command_encoder.clear_buffer(self._get_native_resource(resource))

        # Do the compute pass
        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, self._bind_group)
        compute_pass.dispatch_workgroups(nx, ny, nz)
        compute_pass.end()

        # Submit!
        device.queue.submit([command_encoder.finish()])

        # Timeit
        if self._report_time:
            device._poll_wait()  # wait for the GPU to finish
            t1 = time.perf_counter()
            what = f"Computing {self._label!r}" if self._label else "Computing"
            print(f"{what} took {(t1 - t0) * 1000:0.1f} ms")


## End of compute API


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

# Hard-coded number of bins
nbins = 256

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
# We also set TEXTURE_BINDING because we also want to render the image.
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


# Create a buffer to store bins for the histogram, separately for rgb and luminance.
# The COPY_DST usage is needed to be able to clear the buffer (to zeros).
histogram_bins_buffer = gfx.Buffer(
    nbytes=nbins * 4 * 4,
    nitems=nbins,
    format="4xu4",
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
)

# Create a buffer that holds the line positions. This is written to by a compute shader.
# It consists of 4 pieces (red, green, blue, luminance), with nans in between.
histogram_line_buffer = gfx.Buffer(
    nbytes=(4 * nbins + 3) * 3 * 4,
    nitems=(4 * nbins + 3),
    format="3xf4",
    usage=wgpu.BufferUsage.STORAGE,
)

# Color each line-piece
histogram_colors = np.zeros(((4 * nbins + 3), 3), np.float32)
histogram_colors[0 * nbins + 0 : 1 * nbins + 0] = (1, 0, 0)
histogram_colors[1 * nbins + 1 : 2 * nbins + 1] = (0, 1, 0)
histogram_colors[2 * nbins + 2 : 3 * nbins + 2] = (0, 0, 1)
histogram_colors[3 * nbins + 3 : 4 * nbins + 3] = (1, 1, 1)

# Create the line object with that buffer
histogram_object = gfx.Line(
    gfx.Geometry(positions=histogram_line_buffer, colors=histogram_colors),
    gfx.LineMaterial(color="yellow", color_mode="vertex"),
)
histogram_object.local.y += 10
histogram_object.local.scale_y = 1
histogram_object.local.scale_x = image_texture.size[0] / (nbins - 1)
scene.add(histogram_object)


# Update camera to show the image and histogram
camera.show_object(scene)


# --- Create compute shaders
#
# We use two shaders: one to calculate the histogram and store the
# result in an uint32 buffer, the other to use that buffer to set the
# positions buffer of a line object. Making stuff like this fast is not
# trivial. Basically you want to paralellize as much as you can, and
# avoid locking when multiple threads write to the same memory.

# In the current implementation, this is partially solved by using a
# workgroup, which has its own temporary histogram, so that the
# different workgroups don't get in each-others way. After that. each
# thread in a workgroup is then responsible for adding a single
# histogram value to the final histogram buffer.
#
# The second shader simply copies values from the histogram into the
# positions buffer, where we can use 1024 cores, so it's parellized
# pretty well.
#
# Took inspiration from https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html
# but implemented a slightly simpler variant of its final one.
# It is very likely that a more performant implementation exists. But it's also
# likely that what its the most efficient version depends on the hardware.
#
# About workgroup/invocation id's:
#
# * local_invocation_id: the vec3 indicating the current invocation into the
#   workgroup, as specified using @workgroup_size,  i.e. its position in the
#   workgroup grid.
# * local_invocation_index: the u32 represening the 'flat' local_invocation_id.
# * workgroup_id: the vec3 indicating the position of the workgroup in overall
#   compute shader grid, as specified by dispatch_workgroups().
# * global_invocation_id: workgroup_id * workgroup_size + local_invocation_id.


histogram_wgsl = """

const chunkWidth = 16u;
const chunkHeight = 16u;
const chunkDepth = 4u;

const chunkSize = chunkWidth * chunkHeight * chunkDepth;
const binCount = chunkWidth * chunkHeight;
var<workgroup> bins: array<atomic<u32>, chunkSize>;


@group(0) @binding(0) var imageTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> sa_bins: array<atomic<u32>>;

const kSRGBLuminanceFactors = vec3f(0.2126, 0.7152, 0.0722);
fn srgbLuminance(color: vec3f) -> f32 {
    return saturate(dot(color, kSRGBLuminanceFactors));
}


@compute @workgroup_size(chunkWidth, chunkHeight, chunkDepth)
fn compute_histogram(
    @builtin(global_invocation_id) global_invocation_id: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
) {

    // Write zeros to the workgroup array
    let binIndex = local_invocation_index;
    atomicStore(&bins[binIndex], 0u);

    workgroupBarrier();

    // Collect pixels, and store in workgroup array
    let size = textureDimensions(imageTexture, 0);
    let position = global_invocation_id.xy;
    if (all(position < size)) {
        let lastBinIndex = u32(binCount - 1);
        let color = textureLoad(imageTexture, position, 0);
        let channel = u32(local_invocation_id.z);
        var v: f32;
        if (channel == 3u) {
            v = srgbLuminance(color.rgb);
        } else {
            v = color[channel];
        }
        let binIndex = min(u32(v * f32(binCount)), lastBinIndex) * 4u + channel;
        atomicAdd(&bins[binIndex], 1u);
    }

    workgroupBarrier();

    // Copy the bin values from our workgroup array to the storage buffer.
    let binValue = atomicLoad(&bins[binIndex]);
    if (binValue > 0) {
        atomicAdd(&sa_bins[binIndex], binValue);
    }
}


@group(0) @binding(2) var<storage, read> s_bins: array<u32>;
@group(0) @binding(3) var<storage, read_write> s_positions: array<f32>;

// The scale is hard-coded, but we could add another pass to calculate it
const scale = 1.0 / 100.0;

@compute @workgroup_size(binCount, chunkDepth)
fn write_histogram(
    @builtin(global_invocation_id) global_invocation_id: vec3u,
) {

    let i = global_invocation_id.x; // 0..binCount-1
    let channel = global_invocation_id.y;  // 0..4

    let binValue = s_bins[i* 4 + channel];

    // The line consists of 4 pieces, one for each channel.
    // The '+ channel' is there because we need to put a nan-vertex in between
    // the line pieces.
    let vertex_index = ((channel * binCount) + channel + i) * 3u;

    // Write a nan value between line pieces.
    if (i == 0u && channel > 0u) {
        let nan = bitcast<f32>(0x7fc00000u);  // nan;
        s_positions[vertex_index-3] = nan;
        s_positions[vertex_index-2] = nan;
        s_positions[vertex_index-1] = nan;
    }

    // Write the value
    let ix = vertex_index;
    let iy = vertex_index + 1;
    let iz = vertex_index + 2;
    s_positions[ix] = f32(i);
    s_positions[iy] = f32(binValue) * scale;
    s_positions[iz] = 0.0;
}


"""


histogram_compute = ComputeStep(
    histogram_wgsl,
    entry_point="compute_histogram",
    label="compute_histogram",
    report_time=True,
)
histogram_compute.set_resource(0, image_object.geometry.grid)
histogram_compute.set_resource(1, histogram_bins_buffer, clear=True)

histogram_write = ComputeStep(
    histogram_wgsl,
    entry_point="write_histogram",
    label="write_histogram",
    report_time=True,
)
histogram_write.set_resource(2, histogram_bins_buffer)
histogram_write.set_resource(3, histogram_line_buffer)

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
        size = image_object.geometry.grid.size
        histogram_compute.dispatch(
            int(size[0] / 16 + 0.499999), int(size[1] / 16 + 0.499999)
        )
        histogram_write.dispatch(1)

    renderer.render(scene, camera)
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
