import time
import weakref

import numpy as np
import wgpu.backends.rs

from .. import Renderer, RenderFunctionRegistry
from ...linalg import Matrix4, Vector3
from ...objects import WorldObject
from ...cameras import Camera
from ...resources import Buffer, TextureView
from ...utils import array_from_shadertype

from .postprocessing import RenderTexture, SSAAPostProcessingStep


# Definition uniform struct with standard info related to transforms,
# provided to each shader as uniform at slot 0.
# todo: a combined transform would be nice too, for performance
stdinfo_uniform_type = dict(
    cam_transform=("float32", (4, 4)),
    cam_transform_inv=("float32", (4, 4)),
    projection_transform=("float32", (4, 4)),
    projection_transform_inv=("float32", (4, 4)),
    physical_size=("float32", 2),
    logical_size=("float32", 2),
)


registry = RenderFunctionRegistry()

visibility_all = (
    wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE
)

# Alternative texture formats that we support by padding channels as needed.
# Maps virtual_format -> (wgpu_format, pad_value, nbytes)
ALTTEXFORMAT = {
    "rgb8snorm": ("rgba8snorm", 127, 1),
    "rgb8unorm": ("rgba8unorm", 255, 1),
    "rgb8sint": ("rgba8sint", 127, 1),
    "rgb8uint": ("rgba8uint", 255, 1),
    "rgb16sint": ("rgba16sint", 32767, 2),
    "rgb16uint": ("rgba16uint", 65535, 2),
    "rgb32sint": ("rgba32sint", 2147483647, 4),
    "rgb32uint": ("rgba32uint", 4294967295, 4),
    "rgb16float": ("rgba16float", 1, 2),
    "rgb32float": ("rgba32float", 1, 4),
}


def register_wgpu_render_function(wobject_cls, material_cls):
    """Decorator to register a WGPU render function."""

    def _register_wgpu_renderer(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_wgpu_renderer


class RenderInfo:
    """The type of object passed to each wgpu render function together
    with the world object. Contains stdinfo buffer for now. In time
    will probably also include lights etc.
    """

    def __init__(self, *, stdinfo_uniform):
        self.stdinfo_uniform = stdinfo_uniform


class WgpuRenderer(Renderer):
    """A renderer that renders to a surface."""

    def __init__(self, canvas=None, size=None, pixel_ratio=None, show_fps=False):

        # Check and normalize inputs
        if not (canvas is None or isinstance(canvas, wgpu.gui.WgpuCanvasBase)):
            raise TypeError("The given canvas must be None or a wgpu Canvas.")
        self._canvas = canvas
        if not (size is None or (isinstance(size, (tuple, list)) and len(size) == 2)):
            raise TypeError("The given size must be None or a 2-tuple.")
        self._logical_size = (float(size[0]), float(size[1])) if size else None
        self.pixel_ratio = pixel_ratio
        self._show_fps = bool(show_fps)

        # Do we have enough info?
        if canvas is None and size is None:
            raise ValueError("WgpuRenderer needs either a canvas or a size.")

        # Create adapter and device objects
        self._adapter = wgpu.request_adapter(
            canvas=canvas, power_preference="high-performance"
        )
        self._device = self._adapter.request_device(
            required_features=[], required_limits={}
        )

        self._shader_cache = {}

        # Initialize canvas context
        self._canvas_context = None
        canvas_tex_format = wgpu.TextureFormat.rgba8unorm
        if canvas is not None:
            self._canvas_context = canvas.get_context()
            canvas_tex_format = self._canvas_context.get_preferred_format(self._adapter)
            self._canvas_context.configure(
                device=self._device,
                format=canvas_tex_format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )

        # Prepare render targets. These are placeholders to set during
        # each renderpass, intended to keep together the texture-view
        # object, its size, and its format.
        self._canvas_texture = RenderTexture(canvas_tex_format)
        self._depth_texture = RenderTexture(wgpu.TextureFormat.depth32float)
        # The pick texture has 4 channels, object id, and then 3 more, e.g.
        # the instance nr, vertex nr and weights.
        self._pick_texture = RenderTexture(wgpu.TextureFormat.rgba32sint)
        # We use one or two render textures (for 2+ steps, cycle between the 2)
        self._render_textures = []

        # Prepare post-processing steps. Users can append/insert more
        self._ssa_post_processing_step = SSAAPostProcessingStep(self._device)
        self._postfx = []

        # Prepare other properties
        self._msaa = 1  # todo: cannot set sample_count of render_pass yet

        # Initialize a small buffer to read pixel info into
        # Make it 256 bytes just in case (for bytes_per_row)
        self._pixel_info_buffer = self._device.create_buffer(
            size=256,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        # Keep track of object ids
        self._pick_map = weakref.WeakValueDictionary()

    @property
    def device(self):
        """A reference to the used wgpu device."""
        return self._device

    @property
    def postfx(self):
        """A list of post processing steps. Users can append
        PostProcessingStep objects here to do post-processing of the
        rendered image.

        Warning, this API is provisional.
        """
        # todo: is this the way to go?
        return self._postfx

    @property
    def pixel_ratio(self):
        """Configure the size of the render texture relative to the
        logical size. By default (value is None) the used pixel ratio
        follows the screens pixel ratio on high-res displays, and is 2
        otherwise.

        If the used pixel ratio causes the render texture to be larger
        than the physical size of the canvas, SSAA is applied, resulting
        in a smoother final image with less jagged edges. Alternatively,
        this value can be set to e.g. 0.5 to lower* the resolution (e.g.
        for performance during interaction).
        """
        return self._pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        if value is None:
            self._pixel_ratio = None
        elif isinstance(value, (int, float)):
            self._pixel_ratio = None if value <= 0 else float(value)
        else:
            raise TypeError(
                f"Rendered.pixel_ratio expected None or number, not {value}"
            )

    def render(self, scene: WorldObject, camera: Camera):
        """Main render method, called from the canvas."""

        device = self._device

        now = time.perf_counter()  # noqa
        if self._show_fps:
            if not hasattr(self, "_fps"):
                self._fps = now, now, 1
            elif now > self._fps[0] + 1:
                print(f"FPS: {self._fps[2]/(now - self._fps[0]):0.1f}")
                self._fps = now, now, 1
            else:
                self._fps = self._fps[0], now, self._fps[2] + 1

        # todo: support for alt render pipelines (object that renders to texture then renders that)
        # todo: also note that the fragment shader is (should be) optional
        #      (e.g. depth only passes like shadow mapping or z prepass)

        # Get logical size (as two floats). This size is constant throughout
        # all post-processing render passes.
        if self._logical_size:
            logical_size = self._logical_size
        else:
            logical_size = self._canvas.get_logical_size()
        if not all(x > 0 for x in logical_size):
            return

        # Determine the physical size of the render texture
        if self._canvas:
            canvas_size = self._canvas.get_physical_size()
            canvas_ratio = canvas_size[0] / logical_size[0]
            if self._pixel_ratio:
                pixel_ratio = self._pixel_ratio
            else:
                pixel_ratio = canvas_ratio
                if pixel_ratio <= 1:
                    pixel_ratio = 2.0  # use 2 on non-hidpi displays
        else:
            pixel_ratio = self._pixel_ratio if self._pixel_ratio else 2.0

        # Determine the physical size of the first and last render pass
        scene_size = tuple(max(1, int(pixel_ratio * x)) for x in logical_size)
        final_size = canvas_size if self._canvas else scene_size  # noqa

        # Create more render textures if needed, drop some if we can
        pp_steps = self._postfx
        render_textures = self._render_textures
        render_texture_count = min(2, len(pp_steps) + 1)
        while len(render_textures) < render_texture_count:
            render_textures.append(RenderTexture(wgpu.TextureFormat.rgba8unorm))
        while len(render_textures) > render_texture_count:
            render_textures.pop(-1)

        # Set the size of the textures (is a no-op if the size does not change)
        for t in render_textures:
            t.ensure_size(device, scene_size + (1,))
        self._depth_texture.ensure_size(device, scene_size + (1,))
        self._pick_texture.ensure_size(device, scene_size + (1,))

        # Ensure that matrices are up-to-date
        scene.update_matrix_world()
        camera.set_viewport_size(*logical_size)
        camera.update_matrix_world()  # camera may not be a member of the scene
        camera.update_projection_matrix()

        # Get the list of objects to render (visible and having a material)
        q = self.get_render_list(scene, camera)
        for wobject in q:
            self._pick_map[wobject.id] = wobject

        # Update stdinfo uniform buffer object that we'll use during this render call
        self._update_stdinfo_buffer(camera, scene_size, logical_size)

        # Ensure each wobject has pipeline info
        for wobject in q:
            self._ensure_up_to_date(wobject)

        # Filter out objects that we cannot render
        q = [wobject for wobject in q if wobject._wgpu_pipeline_objects is not None]

        # Render the scene graph (to the first texture)
        command_encoder = device.create_command_encoder()
        self._render_recording(command_encoder, q)
        command_buffers = [command_encoder.finish()]
        device.queue.submit(command_buffers)

        # Render each texture into the next
        for i in range(len(pp_steps)):
            step = pp_steps[i]
            step.render(
                render_textures[0],
                render_textures[1],
            )
            render_textures.insert(0, render_textures.pop(1))  # cycle

        # If we have a canvas, we render into it, applying SSAA if possible
        if self._canvas:
            texture_view_target = self._canvas_context.get_current_texture()
            self._canvas_texture.set_texture_view(texture_view_target)
            self._ssa_post_processing_step.render(
                render_textures[0], self._canvas_texture
            )

    def _render_recording(self, command_encoder, q):

        # You might think that this is slow for large number of world
        # object. But it is actually pretty good. It does iterate over
        # all world objects, and over stuff in each object. But that's
        # it, really.
        # todo: we may be able to speed this up with render bundles though

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for wobject in q:
            wgpu_data = wobject._wgpu_pipeline_objects
            for pinfo in wgpu_data["compute_pipelines"]:
                compute_pass.set_pipeline(pinfo["pipeline"])
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                compute_pass.dispatch(*pinfo["index_args"])

        compute_pass.end_pass()

        # ----- render pipelines rendering to the default target

        assert self._render_textures[0].texture_view
        assert self._depth_texture.texture_view
        assert self._pick_texture.texture_view
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self._render_textures[0].texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                },
                {
                    "view": self._pick_texture.texture_view,
                    "resolve_target": None,
                    "load_value": (0, 0, 0, 0),  # LoadOp.load or color
                    "store_op": wgpu.StoreOp.store,
                },
            ],
            depth_stencil_attachment={
                "view": self._depth_texture.texture_view,
                "depth_load_value": 1.0,  # depth is 0..1, make initial value as high as we can
                "depth_store_op": wgpu.StoreOp.store,
                "stencil_load_value": wgpu.LoadOp.load,
                "stencil_store_op": wgpu.StoreOp.store,
            },
            occlusion_query_set=None,
        )

        for wobject in q:
            wgpu_data = wobject._wgpu_pipeline_objects
            for pinfo in wgpu_data["render_pipelines"]:
                render_pass.set_pipeline(pinfo["pipeline"])
                for slot, vbuffer in pinfo["vertex_buffers"].items():
                    render_pass.set_vertex_buffer(
                        slot,
                        vbuffer._wgpu_buffer[1],
                        vbuffer.vertex_byte_range[0],
                        vbuffer.vertex_byte_range[1],
                    )
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
                # Draw with or without index buffer
                if pinfo["index_buffer"] is not None:
                    ibuffer = pinfo["index_buffer"]
                    render_pass.set_index_buffer(ibuffer, 0, ibuffer.size)
                    render_pass.draw_indexed(*pinfo["index_args"])
                else:
                    render_pass.draw(*pinfo["index_args"])

        render_pass.end_pass()

    def _update_stdinfo_buffer(self, camera, physical_size, logical_size):
        # Make sure we have a buffer object
        if not hasattr(self, "_wgpu_stdinfo_buffer"):
            self._wgpu_stdinfo_buffer = Buffer(
                array_from_shadertype(stdinfo_uniform_type), usage="uniform"
            )

        # Update its data
        stdinfo_data = self._wgpu_stdinfo_buffer.data
        stdinfo_data["cam_transform"].flat = camera.matrix_world_inverse.elements
        stdinfo_data["cam_transform_inv"].flat = camera.matrix_world.elements
        stdinfo_data["projection_transform"].flat = camera.projection_matrix.elements
        stdinfo_data[
            "projection_transform_inv"
        ].flat = camera.projection_matrix_inverse.elements
        # stdinfo_data["ndc_to_world"].flat = np.linalg.inv(stdinfo_data["cam_transform"] @ stdinfo_data["projection_transform"])
        stdinfo_data["physical_size"] = physical_size
        stdinfo_data["logical_size"] = logical_size
        # Upload to GPU
        self._wgpu_stdinfo_buffer.update_range(0, 1)
        self._update_buffer(self._wgpu_stdinfo_buffer)

    def get_render_list(self, scene: WorldObject, camera: Camera):
        """Given a scene object, get a flat list of objects to render."""

        # Collect items
        def visit(wobject):
            nonlocal q
            if wobject.visible and hasattr(wobject, "material"):
                q.append(wobject)

        q = []
        scene.traverse(visit)

        # Next, sort them from back-to-front
        def sort_func(wobject: WorldObject):
            z = (
                Vector3()
                .set_from_matrix_position(wobject.matrix_world)
                .apply_matrix4(proj_screen_matrix)
                .z
            )
            return wobject.render_order, z

        proj_screen_matrix = Matrix4().multiply_matrices(
            camera.projection_matrix, camera.matrix_world_inverse
        )
        q.sort(key=sort_func)
        return q

    def _ensure_up_to_date(self, wobject, force=False):
        """Update the GPU objects associated with the given wobject. Returns
        quickly if no changes are needed.
        """

        # Do we need to create the pipeline infos (from the renderfunc for this wobject)?
        if force or wobject.rev > getattr(wobject, "_wgpu_rev", 0):
            wobject._wgpu_rev = wobject.rev
            wobject._wgpu_pipeline_infos = self._create_pipeline_infos(wobject)
            wobject._wgpu_pipeline_res = self._collect_pipeline_resources(wobject)
            wobject._wgpu_pipeline_objects = None  # Invalidate

        # Early exit?
        if not wobject._wgpu_pipeline_infos:
            return

        # Check if we need to update any resources. The number of
        # resources should typically be small. We could implement a
        # hook in the resource's rev setter so we only have to check
        # one flag ... but let's not optimize prematurely.
        for kind, resource in wobject._wgpu_pipeline_res:
            our_version = getattr(resource, "_wgpu_" + kind, (-1, None))[0]
            if resource.rev > our_version:
                update_func = getattr(self, "_update_" + kind)
                update_func(resource)
                # one of self._update_buffer self._update_texture, self._update_texture_view, self._update_sampler

        # Create gpu objects?
        if wobject._wgpu_pipeline_objects is None:
            wobject._wgpu_pipeline_objects = self._create_pipeline_objects(wobject)

    def _create_pipeline_infos(self, wobject):
        """Use the render function for this wobject and material,
        and return a list of dicts representing pipelines in an abstract way.
        These dicts can then be turned into actual pipeline objects.
        """

        # Get render function for this world object,
        # and use it to get a high-level description of pipelines.
        renderfunc = registry.get_render_function(wobject)
        if renderfunc is None:
            raise ValueError(
                f"Could not get a render function for {wobject.__class__.__name__} "
                f"with {wobject.material.__class__.__name__}"
            )

        # Prepare info for the render function
        render_info = RenderInfo(
            stdinfo_uniform=self._wgpu_stdinfo_buffer,
        )

        # Call render function
        pipeline_infos = renderfunc(wobject, render_info)
        if not pipeline_infos:
            pipeline_infos = None
        else:
            assert isinstance(pipeline_infos, list)

        return pipeline_infos

    def _collect_pipeline_resources(self, wobject):

        pipeline_infos = wobject._wgpu_pipeline_infos or []

        pipeline_resources = []  # List, because order matters

        # Collect list of resources. That we can we can easily iterate over
        # dependent resource on each render call.
        for pipeline_info in pipeline_infos:
            assert isinstance(pipeline_info, dict)
            buffer = pipeline_info.get("index_buffer", None)
            if buffer is not None:
                pipeline_resources.append(("buffer", buffer))
            for buffer in pipeline_info.get("vertex_buffers", {}).values():
                pipeline_resources.append(("buffer", buffer))
            for key in pipeline_info.keys():
                if key.startswith("bindings"):
                    resources = pipeline_info[key]
                    if isinstance(resources, dict):
                        resources = resources.values()
                    for binding_type, resource in resources:
                        if binding_type.startswith("buffer/"):
                            assert isinstance(resource, Buffer)
                            pipeline_resources.append(("buffer", resource))
                        elif binding_type.startswith("sampler/"):
                            assert isinstance(resource, TextureView)
                            pipeline_resources.append(("sampler", resource))
                        elif binding_type.startswith("texture/"):
                            assert isinstance(resource, TextureView)
                            pipeline_resources.append(("texture", resource.texture))
                            pipeline_resources.append(("texture_view", resource))
                        elif binding_type.startswith("storage_texture/"):
                            assert isinstance(resource, TextureView)
                            pipeline_resources.append(("texture", resource.texture))
                            pipeline_resources.append(("texture_view", resource))
                        else:
                            raise RuntimeError(
                                f"Unknown resource binding type {binding_type}"
                            )

        return pipeline_resources

    def _create_pipeline_objects(self, wobject):
        """Generate wgpu pipeline objects from the list of pipeline info dicts."""

        # Prepare the three kinds of pipelines that we can get
        compute_pipelines = []
        render_pipelines = []
        alt_render_pipelines = []

        # Process each pipeline info object, converting each to a more concrete dict
        for pipeline_info in wobject._wgpu_pipeline_infos:
            if "vertex_shader" in pipeline_info and "fragment_shader" in pipeline_info:
                pipeline = self._compose_render_pipeline(wobject, pipeline_info)
                if pipeline_info.get("target", None) is None:
                    render_pipelines.append(pipeline)
                else:
                    raise NotImplementedError("Alternative render pipelines")
                    alt_render_pipelines.append(pipeline)
            elif "compute_shader" in pipeline_info:
                compute_pipelines.append(
                    self._compose_compute_pipeline(wobject, pipeline_info)
                )
            else:
                raise ValueError(
                    "Did not find compute_shader nor vertex_shader+fragment_shader in pipeline info."
                )

        return {
            "compute_pipelines": compute_pipelines,
            "render_pipelines": render_pipelines,
            "alt_render_pipelines": alt_render_pipelines,
        }

    def _compose_compute_pipeline(self, wobject, pipeline_info):
        """Given a high-level compute pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

        # todo: cache the pipeline with the shader (and entrypoint) as a hash

        device = self._device

        # Convert indices to args for the compute_pass.dispatch() call
        indices = pipeline_info["indices"]
        if not (
            isinstance(indices, tuple)
            and len(indices) == 3
            and all(isinstance(i, int) for i in indices)
        ):
            raise RuntimeError(
                f"Compute indices must be 3-tuple of ints, not {indices}."
            )
        index_args = indices

        # Get bind groups and pipeline layout from the buffers in pipeline_info.
        # This also makes sure the buffers and textures are up-to-date.
        bind_groups, pipeline_layout = self._get_bind_groups(pipeline_info)

        # Compile shader and create pipeline object
        cshader = pipeline_info["compute_shader"]
        cs_module = device.create_shader_module(code=cshader)
        compute_pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": cs_module, "entry_point": "main"},
        )

        return {
            "pipeline": compute_pipeline,  # wgpu object
            "index_args": index_args,  # tuple
            "bind_groups": bind_groups,  # list of wgpu bind_group objects
        }

    def _compose_render_pipeline(self, wobject, pipeline_info):
        """Given a high-level render pipeline description, creates a
        lower-level representation that can be consumed by wgpu.
        """

        # todo: cache the pipeline with a lot of things as the hash
        # todo: cache vertex descriptors

        device = self._device

        # If an index buffer is present, update it, and get index_format.
        wgpu_index_buffer = None
        index_format = wgpu.IndexFormat.uint32
        index_buffer = pipeline_info.get("index_buffer", None)
        if index_buffer is not None:
            wgpu_index_buffer = index_buffer._wgpu_buffer[1]
            index_format = index_buffer.format

        # Convert and check high-level indices. Indices represent a range
        # of index id's, or define what indices in the index buffer are used.
        indices = pipeline_info.get("indices", None)
        if indices is None:
            if index_buffer is None:
                raise RuntimeError("Need indices or index_buffer ")
            indices = range(index_buffer.data.size)
        # Convert to 2-element tuple (vertex, instance)
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) == 1:
            indices = indices + (1,)  # add instancing index
        if len(indices) != 2:
            raise RuntimeError("Render pipeline indices must be a 2-element tuple.")

        # Convert indices to args for the render_pass.draw() or draw_indexed()
        # draw(count_vertex, count_instance, first_vertex, first_instance)
        # draw_indexed(count_v, count_i, first_vertex, base_vertex, first_instance)
        index_args = [0, 0, 0, 0]
        for i, index in enumerate(indices):
            if isinstance(index, int):
                index_args[i] = index
            elif isinstance(index, range):
                assert index.step == 1
                index_args[i] = index.stop - index.start
                index_args[i + 2] = index.start
            else:
                raise RuntimeError(
                    "Render pipeline indices must be a 2-element tuple with ints or ranges."
                )
        if wgpu_index_buffer is not None:
            base_vertex = 0  # A value added to each index before reading [...]
            index_args.insert(-1, base_vertex)

        # Process vertex buffers. Update the buffer, and produces a descriptor.
        vertex_buffers = {}
        vertex_buffer_descriptors = []
        # todo: we can probably expose multiple attributes per buffer using a BufferView
        # -> can we also leverage numpy here?
        for slot, buffer in pipeline_info.get("vertex_buffers", {}).items():
            slot = int(slot)
            vbo_des = {
                "array_stride": buffer.nbytes // buffer.nitems,
                "step_mode": wgpu.VertexStepMode.vertex,  # vertex or instance
                "attributes": [
                    {
                        "format": buffer.format,
                        "offset": 0,
                        "shader_location": slot,
                    }
                ],
            }
            vertex_buffers[slot] = buffer
            vertex_buffer_descriptors.append(vbo_des)

        # Get bind groups and pipeline layout from the buffers in pipeline_info.
        # This also makes sure the buffers and textures are up-to-date.
        bind_groups, pipeline_layout = self._get_bind_groups(pipeline_info)

        # Compile shaders
        vs_entry_point = fs_entry_point = "main"
        vshader = pipeline_info["vertex_shader"]
        if isinstance(vshader, tuple):
            vshader, vs_entry_point = vshader
        fshader = pipeline_info["fragment_shader"]
        if isinstance(fshader, tuple):
            fshader, fs_entry_point = fshader
        vs_module = self._get_shader_module(vshader, vshader)
        fs_module = self._get_shader_module(fshader, fshader)

        # Instantiate the pipeline object
        # todo: is this how strip_index_format is supposed to work?
        strip_index_format = 0
        if "strip" in pipeline_info["primitive_topology"]:
            strip_index_format = index_format
        pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": vs_module,
                "entry_point": vs_entry_point,
                "buffers": vertex_buffer_descriptors,
            },
            primitive={
                "topology": pipeline_info["primitive_topology"],
                "strip_index_format": strip_index_format,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": self._depth_texture.format,
                "depth_write_enabled": True,  # optional
                "depth_compare": wgpu.CompareFunction.less,  # optional
                "stencil_front": {},  # use defaults
                "stencil_back": {},  # use defaults
                "depth_bias": 0,
                "depth_bias_slope_scale": 0.0,
                "depth_bias_clamp": 0.0,
            },
            multisample={
                "count": self._msaa,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
            fragment={
                "module": fs_module,
                "entry_point": fs_entry_point,
                "targets": [
                    {
                        "format": self._render_textures[0].format,
                        "blend": {
                            "alpha": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                            "color": (
                                wgpu.BlendFactor.src_alpha,
                                wgpu.BlendFactor.one_minus_src_alpha,
                                wgpu.BlendOperation.add,
                            ),
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    },
                    {
                        "format": self._pick_texture.format,
                        "blend": None,
                        "write_mask": wgpu.ColorWrite.ALL,
                    },
                ],
            },
        )

        return {
            "pipeline": pipeline,  # wgpu object
            "index_args": index_args,  # tuple
            "index_buffer": wgpu_index_buffer,  # Buffer
            "vertex_buffers": vertex_buffers,  # dict of slot -> Buffer
            "bind_groups": bind_groups,  # list of wgpu bind_group objects
        }

    def _get_bind_groups(self, pipeline_info):
        """Given high-level information on bindings, create the corresponding
        wgpu objects. This assumes that all buffers and textures are up-to-date.
        Returns (bind_groups, pipeline_layout).
        """
        # todo: cache bind_group_layout objects
        # todo: cache pipeline_layout objects
        # todo: can perhaps be more specific about visibility

        device = self._device

        # Collect resource groups (keys e.g. "bindings1", "bindings132")
        resource_groups = []
        for key in pipeline_info.keys():
            if key.startswith("bindings"):
                i = int(key[len("bindings") :])
                assert i >= 0
                if not pipeline_info[key]:
                    continue
                while len(resource_groups) <= i:
                    resource_groups.append({})
                resource_groups[i] = pipeline_info[key]

        # Create bind groups and bind group layouts
        bind_groups = []
        bind_group_layouts = []
        for resources in resource_groups:
            if not isinstance(resources, dict):
                resources = {slot: resource for slot, resource in enumerate(resources)}
            # Collect list of dicts
            bindings = []
            binding_layouts = []
            for slot, type_resource in resources.items():
                assert isinstance(type_resource, tuple) and len(type_resource) == 2
                binding_type, resource = type_resource
                subtype = binding_type.split("/")[-1]

                if binding_type.startswith("buffer/"):
                    assert isinstance(resource, Buffer)
                    bindings.append(
                        {
                            "binding": slot,
                            "resource": {
                                "buffer": resource._wgpu_buffer[1],
                                "offset": 0,
                                "size": resource.nbytes,
                            },
                        }
                    )
                    binding_layouts.append(
                        {
                            "binding": slot,
                            "visibility": visibility_all,
                            "buffer": {
                                "type": getattr(wgpu.BufferBindingType, subtype),
                                "has_dynamic_offset": False,
                                "min_binding_size": 0,
                            },
                        }
                    )
                elif binding_type.startswith("sampler/"):
                    assert isinstance(resource, TextureView)
                    bindings.append(
                        {"binding": slot, "resource": resource._wgpu_sampler[1]}
                    )
                    binding_layouts.append(
                        {
                            "binding": slot,
                            "visibility": wgpu.ShaderStage.FRAGMENT,
                            "sampler": {
                                "type": getattr(wgpu.SamplerBindingType, subtype),
                            },
                        }
                    )
                elif binding_type.startswith("texture/"):
                    assert isinstance(resource, TextureView)
                    bindings.append(
                        {"binding": slot, "resource": resource._wgpu_texture_view[1]}
                    )
                    dim = resource.view_dim
                    dim = getattr(wgpu.TextureViewDimension, dim, dim)
                    sample_type = getattr(wgpu.TextureSampleType, subtype, subtype)
                    # Derive sample type from texture
                    if sample_type == "auto":
                        fmt = ALTTEXFORMAT.get(resource.format, [resource.format])[0]
                        if "float" in fmt or "norm" in fmt:
                            sample_type = wgpu.TextureSampleType.float
                            # For float32 wgpu does not allow the sampler to be filterable,
                            # except when the native-only feature
                            # TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is set,
                            # which wgpu-py does by default.
                            # if "32float" in fmt:
                            #     sample_type = wgpu.TextureSampleType.unfilterable_float
                        elif "uint" in fmt:
                            sample_type = wgpu.TextureSampleType.uint
                        elif "sint" in fmt:
                            sample_type = wgpu.TextureSampleType.sint
                        elif "depth" in fmt:
                            sample_type = wgpu.TextureSampleType.depth
                        else:
                            raise ValueError("Could not determine texture sample type.")
                    binding_layouts.append(
                        {
                            "binding": slot,
                            "visibility": wgpu.ShaderStage.FRAGMENT,
                            "texture": {
                                "sample_type": sample_type,
                                "view_dimension": dim,
                                "multisampled": False,
                            },
                        }
                    )
                elif binding_type.startswith("storage_texture/"):
                    assert isinstance(resource, TextureView)
                    bindings.append(
                        {"binding": slot, "resource": resource._wgpu_texture_view[1]}
                    )
                    dim = resource.view_dim
                    dim = getattr(wgpu.TextureViewDimension, dim, dim)
                    fmt = ALTTEXFORMAT.get(resource.format, [resource.format])[0]
                    binding_layouts.append(
                        {
                            "binding": slot,
                            "visibility": visibility_all,
                            "storage_texture": {
                                "access": getattr(wgpu.StorageTextureAccess, subtype),
                                "format": fmt,
                                "view_dimension": dim,
                            },
                        }
                    )

            # Create wgpu objects
            bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
            bind_group = device.create_bind_group(
                layout=bind_group_layout, entries=bindings
            )
            bind_groups.append(bind_group)
            bind_group_layouts.append(bind_group_layout)

        # Create pipeline layout object from list of layouts
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )

        return bind_groups, pipeline_layout

    def _update_buffer(self, resource):
        buffer = getattr(resource, "_wgpu_buffer", (-1, None))[1]

        # todo: dispose an old buffer? / reuse an old buffer?

        pending_uploads = resource._pending_uploads
        resource._pending_uploads = []
        bytes_per_item = resource.nbytes // resource.nitems

        # Create buffer if needed
        if buffer is None or buffer.size != resource.nbytes:
            usage = wgpu.BufferUsage.COPY_DST
            for u in resource.usage.split("|"):
                usage |= getattr(wgpu.BufferUsage, u)
            buffer = self._device.create_buffer(size=resource.nbytes, usage=usage)

        queue = self._device.queue
        encoder = self._device.create_command_encoder()

        # Upload any pending data
        for offset, size in pending_uploads:
            subdata = resource._get_subdata(offset, size)
            # A: map the buffer, writes to it, then unmaps. But we don't offer a mapping API in wgpu-py
            # B: roll data in new buffer, copy from there to existing buffer
            tmp_buffer = self._device.create_buffer_with_data(
                data=subdata,
                usage=wgpu.BufferUsage.COPY_SRC,
            )
            boffset, bsize = bytes_per_item * offset, bytes_per_item * size
            encoder.copy_buffer_to_buffer(tmp_buffer, 0, buffer, boffset, bsize)
            # C: using queue. This may be sugar for B, but it may also be optimized
            # Unfortunately, this seems to crash the device :/
            # queue.write_buffer(buffer, bytes_per_item * offset, subdata, 0, subdata.nbytes)
            # D: A staging buffer/belt https://github.com/gfx-rs/wgpu-rs/blob/master/src/util/belt.rs
            # todo: look into staging buffers?

        queue.submit([encoder.finish()])
        resource._wgpu_buffer = resource.rev, buffer

    def _update_texture_view(self, resource):
        if resource._is_default_view:
            texture_view = resource.texture._wgpu_texture[1].create_view()
        else:
            dim = resource._view_dim
            assert resource._mip_range.step == 1
            assert resource._layer_range.step == 1
            texture_view = resource.texture._wgpu_texture[1].create_view(
                format=ALTTEXFORMAT.get(resource.format, [resource.format])[0],
                dimension=f"{dim}d" if isinstance(dim, int) else dim,
                aspect=resource._aspect,
                base_mip_level=resource._mip_range.start,
                mip_level_count=len(resource._mip_range),
                base_array_layer=resource._layer_range.start,
                array_layer_count=len(resource._layer_range),
            )
        resource._wgpu_texture_view = resource.rev, texture_view

    def _update_texture(self, resource):

        texture = getattr(resource, "_wgpu_texture", (-1, None))[1]
        pending_uploads = resource._pending_uploads
        resource._pending_uploads = []

        format = resource.format
        pixel_padding = None
        if format in ALTTEXFORMAT:
            format, pixel_padding, extra_bytes = ALTTEXFORMAT[format]

        # Create texture if needed
        if texture is None:  # todo: or needs to be replaced (e.g. resized)
            usage = wgpu.TextureUsage.COPY_DST
            for u in resource.usage.split("|"):
                usage |= getattr(wgpu.TextureUsage, u)
            texture = self._device.create_texture(
                size=resource.size,
                usage=usage,
                dimension=f"{resource.dim}d",
                format=getattr(wgpu.TextureFormat, format),
                mip_level_count=1,
                sample_count=1,  # msaa?
            )  # todo: let resource specify mip_level_count and sample_count

        bytes_per_pixel = resource.nbytes // (
            resource.size[0] * resource.size[1] * resource.size[2]
        )
        if pixel_padding is not None:
            bytes_per_pixel += extra_bytes

        queue = self._device.queue
        encoder = self._device.create_command_encoder()

        # Upload any pending data
        for offset, size in pending_uploads:
            subdata = resource._get_subdata(offset, size, pixel_padding)
            # B: using a temp buffer
            # tmp_buffer = self._device.create_buffer_with_data(data=subdata,
            #     usage=wgpu.BufferUsage.COPY_SRC,
            # )
            # encoder.copy_buffer_to_texture(
            #     {
            #         "buffer": tmp_buffer,
            #         "offset": 0,
            #         "bytes_per_row": size[0] * bytes_per_pixel,  # multiple of 256
            #         "rows_per_image": size[1],
            #     },
            #     {
            #         "texture": texture,
            #         "mip_level": 0,
            #         "origin": offset,
            #     },
            #     copy_size=size,
            # )
            # C: using the queue, which may be doing B, but may also be optimized,
            #    and the bytes_per_row limitation does not apply here
            queue.write_texture(
                {"texture": texture, "origin": offset, "mip_level": 0},
                subdata,
                {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
                size,
            )

        queue.submit([encoder.finish()])
        resource._wgpu_texture = resource.rev, texture

    def _update_sampler(self, resource):
        # A sampler's info (and raw object) are stored on a TextureView
        amodes = resource._address_mode.replace(",", " ").split() or ["clamp"]
        while len(amodes) < 3:
            amodes.append(amodes[-1])
        filters = resource._filter.replace(",", " ").split() or ["nearest"]
        while len(filters) < 3:
            filters.append(filters[-1])
        ammap = {"clamp": "clamp-to-edge", "mirror": "mirror-repeat"}
        sampler = self._device.create_sampler(
            address_mode_u=ammap.get(amodes[0], amodes[0]),
            address_mode_v=ammap.get(amodes[1], amodes[1]),
            address_mode_w=ammap.get(amodes[2], amodes[2]),
            mag_filter=filters[0],
            min_filter=filters[1],
            mipmap_filter=filters[2],
            # lod_min_clamp -> use default 0
            # lod_max_clamp -> use default inf
            # compare -> only not-None for comparison samplers!
        )
        resource._wgpu_sampler = resource.rev, sampler

    def _get_shader_module(self, key, source):
        """Compile a shader module object, or re-use it from the cache."""
        # todo: make this work for objects following the ShaderSourceTemplate interface
        # todo: also release shader modules that are no longer used
        if key not in self._shader_cache:
            m = self._device.create_shader_module(code=source)
            self._shader_cache[key] = m
        return self._shader_cache[key]

    # Picking

    def get_pick_info(self, pos):
        """Get information about the given window location. The given
        pos is a 2D point in logical pixels (with the origin at the
        top-left). Returns a dict with fields:

        * "ndc": The position in normalized device coordinates, the 3d element
            being the depth (0..1). Can be translated to the position
            in world coordinates using the camera transforms.
        * "rgba": The value in the color buffer. All zero's when rendering
          directly to the screen (bypassing post-processing).
        * "world_object": the object at that location (provided that
          the object supports picking).
        * Additional pick info may be available, depending on the type of
          object and its material. See the world-object classes for details.
        """

        # Make pos 0..1, so we can scale it to the render texture
        if self._logical_size:
            logical_size = self._logical_size
        else:
            logical_size = self._canvas.get_logical_size()
        float_pos = pos[0] / logical_size[0], pos[1] / logical_size[1]

        can_sample_color = self._render_textures[0].texture is not None

        # Sample
        encoder = self._device.create_command_encoder()
        self._copy_pixel(encoder, self._depth_texture, float_pos, 0)
        if can_sample_color:
            self._copy_pixel(encoder, self._render_textures[0], float_pos, 8)
        self._copy_pixel(encoder, self._pick_texture, float_pos, 16)
        queue = self._device.queue
        queue.submit([encoder.finish()])

        # Collect data from the buffer
        data = self._pixel_info_buffer.map_read()
        depth = data[0:4].cast("f")[0]
        color = tuple(data[8:12].cast("B"))
        pick_value = tuple(data[16:32].cast("i"))
        wobject = self._pick_map.get(pick_value[0], None)
        # Note: the position in world coordinates is not included because
        # it depends on the camera, but we don't "own" the camera.

        info = {
            "ndc": (2 * float_pos[0] - 1, 2 * float_pos[1] - 1, depth),
            "rgba": color if can_sample_color else (0, 0, 0, 0),
            "world_object": wobject,
        }

        if wobject and hasattr(wobject, "material"):
            pick_info = wobject.material._wgpu_get_pick_info(pick_value)
            info.update(pick_info)
        return info

    def _copy_pixel(self, encoder, render_texture, float_pos, buf_offset):

        # Map position to the texture index
        w, h, d = render_texture.size
        x = max(0, min(w - 1, int(float_pos[0] * w)))
        y = max(0, min(h - 1, int(float_pos[1] * h)))

        # Note: bytes_per_row must be a multiple of 256.
        encoder.copy_texture_to_buffer(
            {
                "texture": render_texture.texture,
                "mip_level": 0,
                "origin": (x, y, 0),
            },
            {
                "buffer": self._pixel_info_buffer,
                "offset": buf_offset,
                "bytes_per_row": 256,  # render_texture.bytes_per_pixel,
                "rows_per_image": 1,
            },
            copy_size=(1, 1, 1),
        )

    def snapshot(self, step_index=-1):
        """Create a snapshot of the currently rendered image."""

        # Prepare
        rt = self._render_textures[step_index]
        size = rt.size
        bytes_per_pixel = 4

        # Note, with queue.read_texture the bytes_per_row limitation does not apply.
        data = self._device.queue.read_texture(
            {
                "texture": rt.texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )

        return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)
