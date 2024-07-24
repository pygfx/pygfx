"""
The main renderer class. This class wraps a canvas or texture and it
manages the rendering process.
"""

import time
import weakref

import numpy as np
import wgpu
import pylinalg as la

from ....objects._base import id_provider
from ....objects import (
    KeyboardEvent,
    RootEventHandler,
    PointerEvent,
    WheelEvent,
    WindowEvent,
    WorldObject,
)
from ....cameras import Camera
from ....resources import Texture
from ....resources._base import resource_update_registry
from ....utils import Color
from ....utils.transform import mat_inv  # noqa

from ... import Renderer
from . import blender as blender_module
from .flusher import RenderFlusher
from .pipeline import get_pipeline_container_group
from .update import update_resource, ensure_wgpu_object
from .shared import get_shared
from .environment import get_environment
from .shadowutil import render_shadow_maps
from .mipmapsutil import generate_texture_mipmaps
from .utils import GfxTextureView


def _get_sort_function(camera: Camera):
    """Given a scene object, get a function to sort wobject-tuples"""

    def sort_func(wobject: WorldObject):
        z = la.vec_transform(wobject.world.position, camera.camera_matrix)[2]
        return wobject.render_order, z

    return sort_func


class WgpuRenderer(RootEventHandler, Renderer):
    """Turns Scenes into rasterized images using wgpu.

    The current implementation supports various ``blend_modes`` which control how
    transparency is handled during the rendering process. The following modes exist:

        * "default" or None: Select the default: currently this is "ordered2".
        * "additive": single-pass approach that adds fragments together.
        * "opaque": single-pass approach that ignores transparency.
        * "ordered1": single-pass approach that blends fragments (using alpha
          blending). Can only produce correct results if fragments are drawn
          from back to front.
        * "ordered2": two-pass approach that first processes all opaque
          fragments and then blends transparent fragments (using alpha blending)
          with depth-write disabled. The visual results are usually better than
          ordered1, but still depend on the drawing order.
        * "weighted": two-pass approach for order independent transparency based
          on alpha weights.
        * "weighted_depth": two-pass approach for order independent transparency
          based on alpha weights and depth [1]. Note that the depth range
          affects the (quality of the) visual result.
        * "weighted_plus": three-pass approach for order independent
          transparency, in which the front-most transparent layer is rendered
          correctly, while transparent layers behind it are blended using alpha
          weights.

    Parameters
    ----------
    target : WgpuCanvas or Texture
        The target to render to. It is also used to determine the size of the
        render buffer.
    pixel_ratio : float, optional
        The ratio between the number of pixels in the render buffer versus the
        number of pixels in the display buffer. If None, this will be 1 for
        high-res canvases and 2 otherwise. If greater than 1, SSAA
        (supersampling anti-aliasing) is applied while converting a render
        buffer to a display buffer. If smaller than 1, pixels from the render
        buffer are replicated while converting to a display buffer. This has
        positive performance implications.
    pixel_filter : float, optional
        The strength of the filter when copying the result to the target/canvas.
    show_fps : bool
        Whether to display the frames per second. Beware that
        depending on the GUI toolkit, the canvas may impose a frame rate limit.
    blend_mode : str
        The method for handling transparency. If None, use ``"ordered2"``.
    sort_objects : bool
        If True, sort objects by depth before rendering. The sorting
        uses a hierarchical index based on the object's (1) ``render_order``,
        (2) distance to the camera (based on the local frame's origin), (3) the
        position in the scene graph (flattened depth-first). If False, the
        rendering order is based on the objects ``render_order`` and position
        in the scene graph only.
    enable_events : bool
        If True, forward wgpu events to pygfx's event system.
    gamma_correction : float
        The gamma correction to apply in the final render stage. Typically a
        number between 0.0 and 2.0. A value of 1.0 indicates no correction.

    References
    ----------
    [1] Morgan McGuire and Louis Bavoil, Weighted Blended Order-Independent Transparency, Journal of Computer Graphics Techniques (JCGT), vol. 2, no. 2, 122-141, 2013

    """

    _wobject_pipelines_collection = weakref.WeakValueDictionary()

    def __init__(
        self,
        target,
        *args,
        pixel_ratio=None,
        pixel_filter=None,
        show_fps=False,
        blend_mode="default",
        sort_objects=False,
        enable_events=True,
        gamma_correction=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Check and normalize inputs
        if not isinstance(target, (Texture, GfxTextureView, wgpu.gui.WgpuCanvasBase)):
            raise TypeError(
                f"Render target must be a Canvas or Texture, not a {target.__class__.__name__}"
            )
        self._target = target
        self.pixel_ratio = pixel_ratio
        self.pixel_filter = pixel_filter

        # Make sure we have a shared object (the first renderer creates the instance)
        self._shared = get_shared()
        self._device = self._shared.device

        # Init counter to auto-clear
        self._renders_since_last_flush = 0

        # Get target format
        self.gamma_correction = gamma_correction
        self._gamma_correction_srgb = 1.0
        if isinstance(target, wgpu.gui.WgpuCanvasBase):
            self._canvas_context = self._target.get_context()
            # Select output format. We currently don't have a way of knowing
            # what formats are available, so if not srgb, we gamma-correct in shader.
            target_format = self._canvas_context.get_preferred_format(
                self._shared.adapter
            )
            if not target_format.endswith("srgb"):
                self._gamma_correction_srgb = 1 / 2.2  # poor man's srgb
            # Also configure the canvas
            self._canvas_context.configure(
                device=self._device,
                format=target_format,
            )
        else:
            target_format = self._target.format
            # Also enable the texture for render and display usage
            self._target._wgpu_usage |= wgpu.TextureUsage.RENDER_ATTACHMENT
            self._target._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING

        # Prepare render targets.
        self.blend_mode = blend_mode
        self.sort_objects = sort_objects

        # Prepare object that performs the final render step into a texture
        self._flusher = RenderFlusher(target_format)

        # Initialize a small buffer to read pixel info into
        # Make it 256 bytes just in case (for bytes_per_row)
        self._pixel_info_buffer = self._device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        # Init fps measurements
        self._show_fps = bool(show_fps)
        now = time.perf_counter()
        self._fps = {"start": now, "count": 0}

        if enable_events:
            self.enable_events()

    @property
    def device(self):
        """A reference to the global wgpu device."""
        return self._device

    @property
    def target(self):
        """The render target. Can be a canvas, texture or texture view."""
        return self._target

    @property
    def pixel_ratio(self):
        """The ratio between the number of internal pixels versus the logical pixels on the canvas.

        This can be used to configure the size of the render texture
        relative to the canvas' *logical* size. Can be set to None to
        set the default. By default the pixel_ratio is 2 on "regular"
        screens, and the same as the screen pixel ratio on HiDPI screens
        (usually also 2).

        If the used pixel ratio causes the render texture to be larger
        than the physical size of the canvas, SSAA (super sampling
        antialiasing) is applied, resulting in a smoother final image
        with less jagged edges. Alternatively, this value can be set
        to e.g. 0.5 to *lower* the resolution.
        """
        if self._pixel_ratio is not None:
            return self._pixel_ratio
        elif isinstance(self._target, wgpu.gui.WgpuCanvasBase):
            target_pixel_ratio = self._target.get_pixel_ratio()
            if target_pixel_ratio > 1.0:
                return target_pixel_ratio
        # Default
        return 2.0

    @pixel_ratio.setter
    def pixel_ratio(self, value):
        if not value:
            value = None
        if value is None:
            self._pixel_ratio = None
        elif isinstance(value, (int, float)):
            self._pixel_ratio = abs(float(value))
        else:
            raise TypeError(
                f"Rendered.pixel_ratio expected None or number, not {value}"
            )

    @property
    def pixel_filter(self):
        """The strength of the filter applied to the final pixels.

        The renderer renders everything to an internal texture, which,
        depending on the `pixel_ratio`, may have a differens size than
        the target (i.e. canvas). In the process of rendering the result
        to the target, a filter is applied, resulting in SSAA if the
        size was larger, and a smoothing effect otherwise.

        When the `pixel_filter` is 1.0, the default optimal filter is
        used. Higher values result in more blur. Can be set to 0 to
        disable the filter.
        """
        return self._pixel_filter

    @pixel_filter.setter
    def pixel_filter(self, value):
        if value is None:
            value = 1.0
        self._pixel_filter = max(0.0, float(value))

    @property
    def rect(self):
        """The rectangular viewport for the renderer area."""
        return (0, 0) + self.logical_size

    @property
    def logical_size(self):
        """The size of the render target in logical pixels."""
        target = self._target
        if isinstance(target, wgpu.gui.WgpuCanvasBase):
            return target.get_logical_size()
        elif isinstance(target, Texture):
            return target.size[:2]  # assuming pixel-ratio 1
        else:
            raise TypeError(f"Unexpected render target {target.__class__.__name__}")

    @property
    def physical_size(self):
        """The physical size of the internal render texture."""
        pixel_ratio = self.pixel_ratio
        target_lsize = self.logical_size
        return tuple(max(1, int(pixel_ratio * x)) for x in target_lsize)

    @property
    def blend_mode(self):
        """The method for handling transparency:

        * "default" or None: Select the default: currently this is "ordered2".
        * "additive": single-pass approach that adds fragments together.
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
          in which the front-most transparent layer is rendered correctly, while
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
            "additive": blender_module.AdditiveFragmentBlender,
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
        """Whether to sort world objects by depth before rendering. Default False.

        * ``True``: the render order is defined by 1) the object's ``render_order``
          property; 2) the object's distance to the camera; 3) the position object
          in the scene graph (based on a depth-first search).
        * ``False``: don't sort, the render order is only defined by the
          ``render_order`` and scene graph position.
        """
        return self._sort_objects

    @sort_objects.setter
    def sort_objects(self, value):
        self._sort_objects = bool(value)

    @property
    def gamma_correction(self):
        """The gamma correction applied in the final composition step."""
        return self._gamma_correction

    @gamma_correction.setter
    def gamma_correction(self, value):
        self._gamma_correction = 1.0 if value is None else float(value)
        if isinstance(self._target, wgpu.gui.WgpuCanvasBase):
            self._target.request_draw()

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
        # WeakValueDictionary, if all renderers stop using a certain
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
        rect=None,
        clear_color=None,
        flush=True,
    ):
        """Render a scene with the specified camera as the viewpoint.

        Parameters:
            scene (WorldObject): The scene to render, a WorldObject that
                optionally has child objects.
            camera (Camera): The camera object to use, which defines the
                viewpoint and view transform.
            rect (tuple, optional): The rectangular region to draw into,
                expressed in logical pixels, a.k.a. the viewport.
            clear_color (bool, optional): Whether to clear the color buffer
                before rendering. By default this is True on the first
                call to ``render()`` after a flush, and False otherwise.
            flush (bool, optional): Whether to flush the rendered result into
                the target (texture or canvas). Default True.
        """

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

        # Get size for the render textures
        logical_size = self.logical_size
        physical_size = self.physical_size
        if not all(i > 0 for i in logical_size):
            return
        pixel_ratio = physical_size[1] / logical_size[1]

        # Update the render targets
        self._blender.ensure_target_size(physical_size)

        # Get viewport in physical pixels
        if not rect:
            scene_lsize = logical_size
            scene_psize = physical_size
            physical_viewport = 0, 0, physical_size[0], physical_size[1], 0, 1
        elif len(rect) == 4:
            scene_lsize = rect[2], rect[3]
            physical_viewport = [int(i * pixel_ratio + 0.4999) for i in rect]
            physical_viewport = tuple(physical_viewport) + (0, 1)
            scene_psize = physical_viewport[2], physical_viewport[3]
        else:
            raise ValueError(
                "The viewport rect must be None or 4 elements (x, y, w, h)."
            )

        # Allow objects to prepare just in time. When doing multiple
        # render calls, we don't want to spam. The clear_color flag is
        # a good indicator to detect the first render call.
        if clear_color:
            ev = WindowEvent(
                "before_render",
                target=None,
                root=self,
                width=logical_size[0],
                height=logical_size[1],
                pixel_ratio=self.pixel_ratio,
            )
            self.dispatch_event(ev)

        # Ensure that matrices are up-to-date
        camera.set_view_size(*scene_lsize)
        camera.update_projection_matrix()

        # Prepare the shared object
        self._shared.pre_render_hook()

        # Update stdinfo uniform buffer object that we'll use during this render call
        self._update_stdinfo_buffer(camera, scene_psize, scene_lsize)

        # Get environment
        environment = get_environment(self, scene)

        # Flatten the scenegraph, categorised by render_order
        wobject_dict = {}
        scene.traverse(
            lambda ob: wobject_dict.setdefault(ob.render_order, []).append(ob), True
        )

        # Produce a sorted list of world objects
        wobject_list = []
        if self._sort_objects:
            depth_sort_func = _get_sort_function(camera)
            for render_order in sorted(wobject_dict.keys()):
                wobjects = wobject_dict[render_order]
                wobjects.sort(key=depth_sort_func)
                wobject_list.extend(wobjects)
        else:
            for render_order in sorted(wobject_dict.keys()):
                wobject_list.extend(wobject_dict[render_order])

        # Collect all pipeline container objects
        compute_pipeline_containers = []
        render_pipeline_containers = []
        for wobject in wobject_list:
            if not wobject.material:
                continue
            container_group = get_pipeline_container_group(wobject, environment)
            compute_pipeline_containers.extend(container_group.compute_containers)
            render_pipeline_containers.extend(container_group.render_containers)
            # Enable pipelines to update data on the CPU. This usually includes
            # baking data into buffers. This is CPU intensive, but in practice
            # it is only used by a few materials.
            for func in container_group.bake_functions:
                func(wobject, camera, logical_size)

        # Update *all* buffers and textures that have changed
        for resource in resource_update_registry.get_syncable_resources(flush=True):
            update_resource(resource)

        # Command buffers cannot be reused. If we want some sort of re-use we should
        # look into render bundles. See https://github.com/gfx-rs/wgpu-native/issues/154
        # If we do get this to work, we should trigger a new recording
        # when the wobject's children, visible, render_order, or render_pass changes.

        # Record the rendering of all world objects, or re-use previous recording
        command_buffers = []
        command_buffers += self._render_recording(
            environment,
            wobject_list,
            compute_pipeline_containers,
            render_pipeline_containers,
            physical_viewport,
            clear_color,
        )
        command_buffers += self._blender.perform_combine_pass()

        # Collect commands and submit
        self._device.queue.submit(command_buffers)

        if flush:
            self.flush()

    def flush(self, target=None):
        """Render the result into the target. This method is called
        automatically unless you use ``.render(..., flush=False)``.
        """

        # Print FPS
        now = time.perf_counter()  # noqa
        if self._show_fps:
            if self._fps["count"] == 0:
                print(f"Time to first draw: {now-self._fps['start']:0.2f}")
                self._fps["start"] = now
                self._fps["count"] = 1
            elif now > self._fps["start"] + 1:
                fps = self._fps["count"] / (now - self._fps["start"])
                print(f"FPS: {fps:0.1f}")
                self._fps["start"] = now
                self._fps["count"] = 1
            else:
                self._fps["count"] += 1

        need_mipmaps = False
        if target is None:
            target = self._target

        # Get the wgpu texture view.
        if isinstance(target, wgpu.gui.WgpuCanvasBase):
            wgpu_tex_view = self._canvas_context.get_current_texture().create_view()
        elif isinstance(target, Texture):
            need_mipmaps = target.generate_mipmaps
            wgpu_tex_view = getattr(target, "_wgpu_default_view", None)
            if wgpu_tex_view is None:
                wgpu_tex_view = ensure_wgpu_object(GfxTextureView(target))
                target._wgpu_default_view = wgpu_tex_view
        elif isinstance(target, GfxTextureView):
            need_mipmaps = target.texture.generate_mipmaps
            wgpu_tex_view = ensure_wgpu_object(target)
        else:
            raise TypeError("Unexpected target type.")

        # Reset counter (so we can auto-clear the first next draw)
        self._renders_since_last_flush = 0

        command_buffers = self._flusher.render(
            self._blender.color_view,
            None,
            wgpu_tex_view,
            self._gamma_correction * self._gamma_correction_srgb,
            self._pixel_filter,
        )
        self._device.queue.submit(command_buffers)

        if need_mipmaps:
            generate_texture_mipmaps(target)

    def _render_recording(
        self,
        environment,
        wobject_list,
        compute_pipeline_containers,
        render_pipeline_containers,
        physical_viewport,
        clear_color,
    ):
        # You might think that this is slow for large number of world
        # object. But it is actually pretty good. It does iterate over
        # all world objects, and over stuff in each object. But that's
        # it, really.
        # todo: we may be able to speed this up with render bundles though

        command_encoder = self._device.create_command_encoder()
        blender = self._blender
        if clear_color:
            blender.clear()
        else:
            blender.clear_depth()

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()

        for compute_pipeline_container in compute_pipeline_containers:
            compute_pipeline_container.dispatch(compute_pass, environment)

        compute_pass.end()

        # ----- render pipelines

        # -- process shadow maps
        lights = (
            environment.lights["point_lights"]
            + environment.lights["spot_lights"]
            + environment.lights["directional_lights"]
        )
        render_shadow_maps(lights, wobject_list, command_encoder)

        for pass_index in range(blender.get_pass_count()):
            color_attachments = blender.get_color_attachments(pass_index)
            depth_attachment = blender.get_depth_attachment(pass_index)
            render_mask = blender.passes[pass_index].render_mask
            if not color_attachments:
                continue

            render_pass = command_encoder.begin_render_pass(
                color_attachments=color_attachments,
                depth_stencil_attachment=depth_attachment,
                occlusion_query_set=None,
            )
            render_pass.set_viewport(*physical_viewport)

            for render_pipeline_container in render_pipeline_containers:
                render_pipeline_container.draw(
                    render_pass, environment, pass_index, render_mask
                )

            render_pass.end()

        return [command_encoder.finish()]

    def _update_stdinfo_buffer(self, camera: Camera, physical_size, logical_size):
        # Update the stdinfo buffer's data
        stdinfo_data = self._shared.uniform_buffer.data
        stdinfo_data["cam_transform"] = camera.world.inverse_matrix.T
        stdinfo_data["cam_transform_inv"] = camera.world.matrix.T
        stdinfo_data["projection_transform"] = camera.projection_matrix.T
        stdinfo_data["projection_transform_inv"] = camera.projection_matrix_inverse.T
        # stdinfo_data["ndc_to_world"].flat = mat_inv(stdinfo_data["cam_transform"] @ stdinfo_data["projection_transform"])
        stdinfo_data["physical_size"] = physical_size
        stdinfo_data["logical_size"] = logical_size
        # Upload to GPU
        self._shared.uniform_buffer.update_full()

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
        logical_size = self.logical_size
        float_pos = pos[0] / logical_size[0], pos[1] / logical_size[1]

        # Prevent out of range and picking before first draw
        out_of_range = not (0 <= float_pos[0] <= 1 and 0 <= float_pos[1] <= 1)
        blender_zero_size = self._blender.size == (0, 0)
        if out_of_range or blender_zero_size:
            return {"rgba": Color(0, 0, 0, 0), "world_object": None}

        # Sample
        encoder = self._device.create_command_encoder()
        self._copy_pixel(encoder, self._blender.color_tex, float_pos, 0)
        self._copy_pixel(encoder, self._blender.pick_tex, float_pos, 8)
        self._device.queue.submit([encoder.finish()])

        # Collect data from the buffer
        self._pixel_info_buffer.map("read")
        try:
            data = self._pixel_info_buffer.read_mapped()
        finally:
            self._pixel_info_buffer.unmap()
        color = Color(x / 255 for x in tuple(data[0:4].cast("B")))
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
        texture = self._blender.color_tex
        size = texture.size
        bytes_per_pixel = 4

        # Note, with queue.read_texture the bytes_per_row limitation does not apply.
        data = self._device.queue.read_texture(
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

    def request_draw(self, draw_function=None):
        """Forwards a request_draw call to the target canvas. If the renderer's
        target is not a canvas (e.g. a texture) this function does
        nothing.
        """
        request_draw = getattr(self.target, "request_draw", None)
        if request_draw:
            request_draw(draw_function)

    def enable_events(self):
        """Add event handlers for a specific list of events that are generated
        by the canvas. The handler is the ``convert_event`` method in order to
        convert the Wgpu event dicts into Pygfx event objects."""

        # Check for ``add_event_handler`` attribute. Silently 'fail' as it is
        # perfectly fine to pass a texture as a target to the renderer
        if hasattr(self._target, "add_event_handler"):
            self._target.add_event_handler(self.convert_event, *EVENTS_TO_CONVERT)

    def disable_events(self):
        """Remove the event handlers from the canvas."""
        if hasattr(self._target, "remove_event_handler"):
            self._target.remove_event_handler(self.convert_event, *EVENTS_TO_CONVERT)

    def convert_event(self, event: dict):
        """Converts Wgpu event (dict following jupyter_rfb spec) to Pygfx Event object,
        adds picking info and then dispatches the event in the Pygfx event system.
        """
        event_type = event["event_type"]
        target = None
        if "x" in event and "y" in event:
            info = self.get_pick_info((event["x"], event["y"]))
            target = info["world_object"]
            event["pick_info"] = info

        ev = EVENT_TYPE_MAP[event_type](
            type=event_type, **event, target=target, root=self
        )
        self.dispatch_event(ev)


EVENT_TYPE_MAP = {
    "resize": WindowEvent,
    "close": WindowEvent,
    "pointer_down": PointerEvent,
    "pointer_up": PointerEvent,
    "pointer_move": PointerEvent,
    "double_click": PointerEvent,
    "wheel": WheelEvent,
    "key_down": KeyboardEvent,
    "key_up": KeyboardEvent,
}


EVENTS_TO_CONVERT = (
    "key_down",
    "key_up",
    "pointer_down",
    "pointer_move",
    "pointer_up",
    "wheel",
    "close",
    "resize",
)
