import time
import weakref

import numpy as np
import wgpu.backends.rs

from .. import Renderer
from ...linalg import Matrix4, Vector3
from ...objects import WorldObject, id_provider
from ...cameras import Camera
from ...resources import Buffer, Texture, TextureView
from ...utils import array_from_shadertype

from . import _blender as blender_module
from ._flusher import RenderFlusher
from ._pipelinebuilder import ensure_pipeline
from ._update import update_buffer, update_texture, update_texture_view


# Definition uniform struct with standard info related to transforms,
# provided to each shader as uniform at slot 0.
# todo: a combined transform would be nice too, for performance
# todo: same for ndc_to_world transform (combined inv transforms)
stdinfo_uniform_type = dict(
    cam_transform="4x4xf4",
    cam_transform_inv="4x4xf4",
    projection_transform="4x4xf4",
    projection_transform_inv="4x4xf4",
    physical_size="2xf4",
    logical_size="2xf4",
    flipped_winding="i4",  # A bool, really
)


def get_size_from_render_target(target):
    """Get physical and logical size from a render target."""
    if isinstance(target, wgpu.gui.WgpuCanvasBase):
        physical_size = target.get_physical_size()
        logical_size = target.get_logical_size()
    elif isinstance(target, Texture):
        physical_size = target.size[:2]
        logical_size = physical_size
    elif isinstance(target, TextureView):
        physical_size = target.texture.size[:2]
        logical_size = physical_size
    else:
        raise TypeError(f"Unexpected render target {target.__class__.__name__}")
    return physical_size, logical_size


def _get_sort_function(camera: Camera):
    """Given a scene object, get a function to sort wobject-tuples"""

    def sort_func(wobject_tuple: WorldObject):
        wobject = wobject_tuple[0]
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

    return sort_func


class SharedData:
    """An object to store global data to share between multiple wgpu renderers.

    Since renderers don't render simultaneously, they can share certain
    resources. This safes memory, but more importantly, resources that
    get used in wobject pipelines should be shared to avoid having to
    constantly recompose the pipelines of wobjects that are rendered by
    multiple renderers.
    """

    def __init__(self, canvas):

        # Create adapter and device objects - there should be just one per canvas.
        # Having a global device provides the benefit that we can draw any object
        # anywhere.
        # We could pass the canvas to request_adapter(), so we get an adapter that is
        # at least compatible with the first canvas that a renderer is create for.
        # However, passing the object has been shown to prevent the creation of
        # a canvas (on Linux + wx), so, we never pass it for now.
        self.adapter = wgpu.request_adapter(
            canvas=None, power_preference="high-performance"
        )
        self.device = self.adapter.request_device(
            required_features=[], required_limits={}
        )

        # Create a uniform buffer for std info
        self.stdinfo_buffer = Buffer(array_from_shadertype(stdinfo_uniform_type))
        self.stdinfo_buffer._wgpu_usage |= wgpu.BufferUsage.UNIFORM

        # A cache for shader objects
        self.shader_cache = {}


class WgpuRenderer(Renderer):
    """Object used to render scenes using wgpu.

    The purpose of a renderer is to render (i.e. draw) a scene to a
    canvas or texture. It also provides picking, defines the
    anti-aliasing parameters, and any post processing effects.

    A renderer is directly associated with its target and can only render
    to that target. Different renderers can render to the same target though.

    It provides a ``.render()`` method that can be called one or more
    times to render scenes. This creates a visual representation that
    is stored internally, and is finally rendered into its render target
    (the canvas or texture).
                                  __________
                                 | blender  |
        [scenes] -- render() --> |  state   | -- flush() --> [target]
                                 |__________|

    The internal representation is managed by the blender object. The
    internal render textures are typically at a higher resolution to
    reduce aliasing (SSAA). The blender has auxilary buffers such as a
    depth buffer, pick buffer, and buffers for transparent fragments.
    Depending on the blend mode, a single render call may consist of
    multiple passes (to deal with semi-transparent fragments).

    The flush-step resolves the internal representation into the target
    texture or canvas, averaging neighbouring fragments for anti-aliasing.

    Parameters:
        target (WgpuCanvas or Texture): The target to render to, and what
            determines the size of the render buffer.
        pixel_ratio (float, optional): How large the physical size of the render
            buffer is in relation to the target's physical size, for antialiasing.
            See the corresponding property for details.
        show_fps (bool): Whether to display the frames per second. Beware that
            depending on the GUI toolkit, the canvas may impose a frame rate limit.
    """

    _shared = None

    _wobject_pipelines_collection = weakref.WeakValueDictionary()

    def __init__(
        self,
        target,
        *,
        pixel_ratio=None,
        show_fps=False,
        blend_mode="default",
        sort_objects=False,
    ):

        # Check and normalize inputs
        if not isinstance(target, (Texture, TextureView, wgpu.gui.WgpuCanvasBase)):
            raise TypeError(
                f"Render target must be a canvas or texture (view), not a {target.__class__.__name__}"
            )
        self._target = target

        # Process other inputs
        self.pixel_ratio = pixel_ratio
        self._show_fps = bool(show_fps)

        # Make sure we have a shared object (the first renderer create it)
        canvas = target if isinstance(target, wgpu.gui.WgpuCanvasBase) else None
        if WgpuRenderer._shared is None:
            WgpuRenderer._shared = SharedData(canvas)

        # Init counter to auto-clear
        self._renders_since_last_flush = 0

        # Get target format
        if isinstance(target, wgpu.gui.WgpuCanvasBase):
            self._canvas_context = self._target.get_context()
            self._target_tex_format = self._canvas_context.get_preferred_format(
                self._shared.adapter
            )
            # Also configure the canvas
            self._canvas_context.configure(
                device=self._shared.device,
                format=self._target_tex_format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
        else:
            self._target_tex_format = self._target.format
            # Also enable the texture for render and display usage
            self._target._wgpu_usage |= wgpu.TextureUsage.RENDER_ATTACHMENT
            self._target._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING

        # Prepare render targets.
        self.blend_mode = blend_mode
        self.sort_objects = sort_objects

        # Prepare object that performs the final render step into a texture
        self._flusher = RenderFlusher(self._shared.device)

        # Initialize a small buffer to read pixel info into
        # Make it 256 bytes just in case (for bytes_per_row)
        self._pixel_info_buffer = self._shared.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    @property
    def device(self):
        """A reference to the used wgpu device."""
        return self._shared.device

    @property
    def target(self):
        """The render target. Can be a canvas, texture or texture view."""
        return self._target

    @property
    def pixel_ratio(self):
        """The ratio between the number of internal pixels versus the logical pixels on the canvas.

        This can be used to configure the size of the render texture
        relative to the canvas' logical size. By default (value is None) the
        used pixel ratio follows the screens pixel ratio on high-res
        displays, and is 2 otherwise.

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

    @property
    def blend_mode(self):
        """The method for handling transparency:

        * "default" or None: Select the default: currently this is "ordered2".
        * "opaque": single-pass approach that consider every fragment opaque.
        * "ordered1": single-pass approach that blends fragments (using alpha blending).
          Can only produce correct results if fragments are drawn from back to front.
        * "ordered2": two-pass approach that first processes all opaque fragments and then
          blends transparent fragments (using alpha blending) with depth-write disabled. The
          visual results are usually better than ordered1, but still depend on the drawing order.
        * "weighted": two-pass approach that for order independent transparency,
          using alpha weights.
        * "weighted_depth": two-pass approach for order independent transparency,
          with weights based on alpha and depth (McGuire 2013). Note that the depth
          range affects the (quality of the) visual result.
        * "weighted_plus": three-pass approach for order independent transparency,
          in wich the front-most transparent layer is rendered correctly, while
          transparent layers behind it are blended using alpha weights.
        """
        return self._blend_mode

    @blend_mode.setter
    def blend_mode(self, value):
        # Massage and check the input
        if value is None:
            value = "default"
        value = value.lower()
        if value == "default":
            value = "ordered2"
        # Map string input to a class
        m = {
            "opaque": blender_module.OpaqueFragmentBlender,
            "ordered1": blender_module.Ordered1FragmentBlender,
            "ordered2": blender_module.Ordered2FragmentBlender,
            "weighted": blender_module.WeightedFragmentBlender,
            "weighted_depth": blender_module.WeightedDepthFragmentBlender,
            "weighted_plus": blender_module.WeightedPlusFragmentBlender,
        }
        if value not in m:
            raise ValueError(
                f"Unknown blend_mode '{value}', use any of {set(m.keys())}"
            )
        # Set blender object
        self._blend_mode = value
        self._blender = m[value]()
        # If the blend mode has changed, we may need a new _wobject_pipelines
        self._set_wobject_pipelines()
        # If our target is a canvas, request a new draw
        if isinstance(self._target, wgpu.gui.WgpuCanvasBase):
            self._target.request_draw()

    @property
    def sort_objects(self):
        """Whether to sort world objects before rendering. Default False.

        * ``True``: the render order is defined by 1) the object's ``render_order``
          property; 2) the object's distance to the camera; 3) the position object
          in the scene graph (based on a depth-first search).
        * ``False``: don't sort, the render order is defined by the scene graph alone.
        """
        return self._sort_objects

    @sort_objects.setter
    def sort_objects(self, value):
        self._sort_objects = bool(value)

    def _set_wobject_pipelines(self):
        # Each WorldObject has associated with it a wobject_pipeline:
        # a dict that contains the wgpu pipeline objects. This
        # wobject_pipeline is also associated with the blend_mode,
        # because the blend mode affects the pipelines.
        #
        # Each renderer has ._wobject_pipelines, a dict that maps
        # wobject -> wobject_pipeline. This dict is a WeakKeyDictionary -
        # when the wobject is destroyed, the associated pipeline is
        # collected as well.
        #
        # Renderers with the same blend mode can safely share these
        # wobject_pipeline dicts. Therefore, we make use of a global
        # collection. Since this global collection is a
        # WeakValueDictionary, if all renderes stop using a certain
        # blend mode, the associated pipelines are removed as well.
        #
        # In a diagram:
        #
        # _wobject_pipelines_collection -> _wobject_pipelines -> wobject_pipeline
        #        global                         renderer              wobject
        #   WeakValueDictionary              WeakKeyDictionary         dict

        # Below we set this renderer's _wobject_pipelines. Note that if the
        # blending has changed, we automatically invalidate all "our" pipelines.
        self._wobject_pipelines = WgpuRenderer._wobject_pipelines_collection.setdefault(
            self.blend_mode, weakref.WeakKeyDictionary()
        )

    def render(
        self,
        scene: WorldObject,
        camera: Camera,
        *,
        viewport=None,
        clear_color=None,
        flush=True,
    ):
        """Render a scene with the specified camera as the viewpoint.

        Parameters:
            scene (WorldObject): The scene to render, a WorldObject that
                optionally has child objects.
            camera (Camera): The camera object to use, which defines the
                viewpoint and view transform.
            viewport (tuple, optional): The rectangular region to draw into,
                expressed in logical pixels.
            clear_color (bool, optional): Whether to clear the color buffer
                before rendering. By default this is True on the first
                call to ``render()`` after a flush, and False otherwise.
            flush (bool, optional): Whether to flush the rendered result into
                the target (texture or canvas). Default True.
        """
        device = self.device

        now = time.perf_counter()  # noqa
        if self._show_fps:
            if not hasattr(self, "_fps"):
                self._fps = now, now, 1
            elif now > self._fps[0] + 1:
                print(f"FPS: {self._fps[2]/(now - self._fps[0]):0.1f}")
                self._fps = now, now, 1
            else:
                self._fps = self._fps[0], now, self._fps[2] + 1

        # Define whether to clear color.
        if clear_color is None:
            clear_color = self._renders_since_last_flush == 0
        clear_color = bool(clear_color)
        self._renders_since_last_flush += 1

        # We always clear the depth, because each render() should be "self-contained".
        # Any use-cases where you normally would control depth-clearing should
        # be covered by the blender. Also, this way the blender can better re-use internal
        # buffers. The only rule is that the color buffer behaves correctly on multiple renders.

        # todo: also note that the fragment shader is (should be) optional
        #      (e.g. depth only passes like shadow mapping or z prepass)

        # Get logical size (as two floats). This size is constant throughout
        # all post-processing render passes.
        target_size, logical_size = get_size_from_render_target(self._target)
        if not all(x > 0 for x in logical_size):
            return

        # Determine the physical size of the render texture
        target_pixel_ratio = target_size[0] / logical_size[0]
        if self._pixel_ratio:
            pixel_ratio = self._pixel_ratio
        else:
            pixel_ratio = target_pixel_ratio
            if pixel_ratio <= 1:
                pixel_ratio = 2.0  # use 2 on non-hidpi displays

        # Determine the physical size of the first and last render pass
        framebuffer_size = tuple(max(1, int(pixel_ratio * x)) for x in logical_size)

        # Update the render targets
        self._blender.ensure_target_size(device, framebuffer_size)

        # Get viewport in physical pixels
        if not viewport:
            scene_logical_size = logical_size
            scene_physical_size = framebuffer_size
            physical_viewport = 0, 0, framebuffer_size[0], framebuffer_size[1], 0, 1
        elif len(viewport) == 4:
            scene_logical_size = viewport[2], viewport[3]
            physical_viewport = [int(i * pixel_ratio + 0.4999) for i in viewport]
            physical_viewport = tuple(physical_viewport) + (0, 1)
            scene_physical_size = physical_viewport[2], physical_viewport[3]
        else:
            raise ValueError("The viewport must be None or 4 elements (x, y, w, h).")

        # Ensure that matrices are up-to-date
        scene.update_matrix_world()
        camera.set_view_size(*scene_logical_size)
        camera.update_matrix_world()  # camera may not be a member of the scene
        camera.update_projection_matrix()

        # Update stdinfo uniform buffer object that we'll use during this render call
        self._update_stdinfo_buffer(camera, scene_physical_size, scene_logical_size)

        # Get the list of objects to render, as they appear in the scene graph
        wobject_list = []
        scene.traverse(wobject_list.append, True)

        # Ensure each wobject has pipeline info, and filter objects that we cannot render
        wobject_tuples = []
        any_has_changed = False
        for wobject in wobject_list:
            if not wobject.material:
                continue
            wobject_pipeline, has_changed = ensure_pipeline(self, wobject)
            if wobject_pipeline:
                any_has_changed |= has_changed
                wobject_tuples.append((wobject, wobject_pipeline))

        # Command buffers cannot be reused. If we want some sort of re-use we should
        # look into render bundles. See https://github.com/gfx-rs/wgpu-native/issues/154
        # If we do get this to work, we should trigger a new recording
        # when the wobject's children, visibile, render_order, or render_pass changes.

        # Sort objects
        if self.sort_objects:
            sort_func = _get_sort_function(camera)
            wobject_tuples.sort(key=sort_func)

        # Record the rendering of all world objects, or re-use previous recording
        command_buffers = []
        command_buffers += self._render_recording(
            wobject_tuples, physical_viewport, clear_color
        )
        command_buffers += self._blender.perform_combine_pass(self._shared.device)
        command_buffers

        # Collect commands and submit
        device.queue.submit(command_buffers)

        if flush:
            self.flush()

    def flush(self):
        """Render the result into the target texture view. This method is
        called automatically unless you use ``.render(..., flush=False)``.
        """

        # Note: we could, in theory, allow specifying a custom target here.

        if isinstance(self._target, wgpu.gui.WgpuCanvasBase):
            raw_texture_view = self._canvas_context.get_current_texture()
        else:
            if isinstance(self._target, Texture):
                texture_view = self._target.get_view()
            elif isinstance(self._target, TextureView):
                texture_view = self._target
            update_texture(self._shared.device, texture_view.texture)
            update_texture_view(self._shared.device, texture_view)
            raw_texture_view = texture_view._wgpu_texture_view[1]

        # Reset counter (so we can auto-clear the first next draw)
        self._renders_since_last_flush = 0

        command_buffers = self._flusher.render(
            self._blender.color_view,
            None,
            raw_texture_view,
            self._target_tex_format,
        )
        self.device.queue.submit(command_buffers)

    def _render_recording(
        self,
        wobject_tuples,
        physical_viewport,
        clear_color,
    ):

        # You might think that this is slow for large number of world
        # object. But it is actually pretty good. It does iterate over
        # all world objects, and over stuff in each object. But that's
        # it, really.
        # todo: we may be able to speed this up with render bundles though

        command_encoder = self.device.create_command_encoder()
        blender = self._blender

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for wobject, wobject_pipeline in wobject_tuples:
            for pinfo in wobject_pipeline.get("compute_pipelines", ()):
                compute_pass.set_pipeline(pinfo["pipeline"])
                for bind_group_id, bind_group in enumerate(pinfo["bind_groups"]):
                    compute_pass.set_bind_group(
                        bind_group_id, bind_group, [], 0, 999999
                    )
                compute_pass.dispatch(*pinfo["index_args"])

        compute_pass.end_pass()

        # ----- render pipelines

        for pass_index in range(blender.get_pass_count()):

            color_attachments = blender.get_color_attachments(pass_index, clear_color)
            depth_attachment = blender.get_depth_attachment(pass_index)
            render_mask = blender.passes[pass_index].render_mask
            if not color_attachments:
                continue

            render_pass = command_encoder.begin_render_pass(
                color_attachments=color_attachments,
                depth_stencil_attachment={
                    **depth_attachment,
                    "stencil_load_value": wgpu.LoadOp.load,
                    "stencil_store_op": wgpu.StoreOp.store,
                },
                occlusion_query_set=None,
            )
            render_pass.set_viewport(*physical_viewport)

            for wobject, wobject_pipeline in wobject_tuples:
                if not (render_mask & wobject_pipeline["render_mask"]):
                    continue
                for pinfo in wobject_pipeline["render_pipelines"]:
                    render_pass.set_pipeline(pinfo["pipelines"][pass_index])
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

        return [command_encoder.finish()]

    def _update_stdinfo_buffer(self, camera, physical_size, logical_size):
        # Update the stdinfo buffer's data
        stdinfo_data = self._shared.stdinfo_buffer.data
        stdinfo_data["cam_transform"].flat = camera.matrix_world_inverse.elements
        stdinfo_data["cam_transform_inv"].flat = camera.matrix_world.elements
        stdinfo_data["projection_transform"].flat = camera.projection_matrix.elements
        stdinfo_data[
            "projection_transform_inv"
        ].flat = camera.projection_matrix_inverse.elements
        # stdinfo_data["ndc_to_world"].flat = np.linalg.inv(stdinfo_data["cam_transform"] @ stdinfo_data["projection_transform"])
        stdinfo_data["physical_size"] = physical_size
        stdinfo_data["logical_size"] = logical_size
        stdinfo_data["flipped_winding"] = camera.flips_winding
        # Upload to GPU
        self._shared.stdinfo_buffer.update_range(0, 1)
        update_buffer(self._shared.device, self._shared.stdinfo_buffer)

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
        _, logical_size = get_size_from_render_target(self._target)
        float_pos = pos[0] / logical_size[0], pos[1] / logical_size[1]

        # Sample
        encoder = self.device.create_command_encoder()
        self._copy_pixel(encoder, self._blender.color_tex, float_pos, 0)
        self._copy_pixel(encoder, self._blender.pick_tex, float_pos, 8)
        queue = self.device.queue
        queue.submit([encoder.finish()])

        # Collect data from the buffer
        data = self._pixel_info_buffer.map_read()
        color = tuple(data[0:4].cast("B"))
        pick_value = tuple(data[8:16].cast("Q"))[0]
        wobject_id = pick_value & 1048575  # 2**20-1
        wobject = id_provider.get_object_from_id(wobject_id)
        # Note: the position in world coordinates is not included because
        # it depends on the camera, but we don't "own" the camera.

        info = {
            "rgba": color,
            "world_object": wobject,
        }

        if wobject:
            pick_info = wobject._wgpu_get_pick_info(pick_value)
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
                "texture": render_texture,
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

    def snapshot(self):
        """Create a snapshot of the currently rendered image."""

        # Prepare
        device = self._shared.device
        texture = self._blender.color_tex
        size = texture.size
        bytes_per_pixel = 4

        # Note, with queue.read_texture the bytes_per_row limitation does not apply.
        data = device.queue.read_texture(
            {
                "texture": texture,
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
