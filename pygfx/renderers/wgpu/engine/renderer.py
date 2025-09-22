"""
The main renderer class. This class wraps a canvas or texture and it
manages the rendering process.
"""

import os
import time
import logging
from typing import Literal

import numpy as np
import wgpu
import pylinalg as la
from rendercanvas import BaseRenderCanvas

from ....objects._base import id_provider
from ....objects import (
    KeyboardEvent,
    RootEventHandler,
    PointerEvent,
    WheelEvent,
    WindowEvent,
    WorldObject,
)
from ....objects._lights import (
    Light,
    PointLight,
    DirectionalLight,
    SpotLight,
    AmbientLight,
)
from ....cameras import Camera
from ....resources import Texture
from ....resources._base import resource_update_registry
from ....utils import Color
from ....utils.enums import PixelFilter

from ... import Renderer
from .blender import Blender
from .effectpasses import EffectPass, OutputPass, PPAAPass, FXAAPass, DDAAPass
from .pipeline import get_pipeline_container_group
from .update import update_resource, ensure_wgpu_object
from .shared import get_shared
from .renderstate import get_renderstate
from .shadowutil import render_shadow_maps
from .mipmapsutil import generate_texture_mipmaps
from .utils import GfxTextureView


logger = logging.getLogger("pygfx")


class WobjectWrapper:
    """To temporary wrap each wobject for each draw."""

    __slots__ = ["pass_type", "render_containers", "sort_key", "wobject"]

    def __init__(self, wobject, sort_key, pass_type):
        self.wobject = wobject
        self.sort_key = sort_key
        self.pass_type = pass_type
        self.render_containers = None


class FlatScene:
    def __init__(self, scene, view_matrix=None, object_count=0):
        self._view_matrix = view_matrix
        self._wobject_wrappers = []  # WobjectWrapper's
        self.lights = {
            "point_lights": [],
            "directional_lights": [],
            "spot_lights": [],
            "ambient_color": [0, 0, 0],
        }
        self.shadow_objects = []
        self.object_count = object_count
        self.add_scene(scene)

    def _iter_scene(self, ob, render_order=0):
        if not ob.visible:
            return
        render_order += ob.render_order
        yield ob, render_order
        for child in ob._children:
            yield from self._iter_scene(child, render_order)

    def add_scene(self, scene):
        """Add a scene to the total flat scene. Is usually called just once."""

        # Put some attributes as vars in this namespace for faster access
        view_matrix = self._view_matrix
        wobject_wrappers = self._wobject_wrappers

        for wobject, render_order in self._iter_scene(scene):
            # Dereference the object in case its a weak proxy
            wobject = wobject._self()
            # Assign renderer id's
            self.object_count += wobject._assign_renderer_id(self.object_count + 1)
            # Update things like transform and uniform buffers
            wobject._update_object()

            # Light objects
            if isinstance(wobject, Light):
                if isinstance(wobject, PointLight):
                    self.lights["point_lights"].append(wobject)
                elif isinstance(wobject, DirectionalLight):
                    self.lights["directional_lights"].append(wobject)
                elif isinstance(wobject, SpotLight):
                    self.lights["spot_lights"].append(wobject)
                elif isinstance(wobject, AmbientLight):
                    r, g, b = wobject.color.to_physical()
                    ambient_color = self.lights["ambient_color"]
                    ambient_color[0] += r * wobject.intensity
                    ambient_color[1] += g * wobject.intensity
                    ambient_color[2] += b * wobject.intensity

            # Shadowable objects
            if wobject.cast_shadow and wobject.geometry is not None:
                self.shadow_objects.append(wobject)

            # Renderable objects
            material = wobject._material
            if material is not None:
                render_queue = material.render_queue
                alpha_method = material.alpha_method

                # By default sort back-to-front, for correct blending.
                dist_sort_sign = -1
                # But for opaque queues, render front-to-back to avoid overdraw.
                if 1500 < render_queue <= 2500:
                    dist_sort_sign = 1

                pass_type = alpha_method

                # Get depth sorting flag. Note that use camera's view matrix, since the projection does not affect the depth order.
                # It also means we can set projection=False optimalization.
                # Also note that we look down -z.
                if alpha_method == "weighted":
                    dist_flag = 0
                elif view_matrix is None:
                    dist_flag = -1
                else:
                    relative_pos = la.vec_transform(
                        wobject.world.position, view_matrix, projection=False
                    )
                    # Cam looks towards -z: negate to get distance
                    distance_to_camera = float(-relative_pos[2])
                    dist_flag = distance_to_camera * dist_sort_sign

                sort_key = (render_queue, render_order, dist_flag)
                wobject_wrappers.append(WobjectWrapper(wobject, sort_key, pass_type))

    def sort(self):
        """Sort the world objects."""
        self._wobject_wrappers.sort(key=lambda ob: ob.sort_key)

    def collect_pipelines_container_groups(self, renderstate):
        """Select and resolve the pipeline, compiling shaders, building pipelines and composing binding as needed."""
        self._compute_pipeline_containers = compute_pipeline_containers = []
        self._bake_functions = bake_functions = []
        for wrapper in self._wobject_wrappers:
            container_group = get_pipeline_container_group(wrapper.wobject, renderstate)
            compute_pipeline_containers.extend(container_group.compute_containers)
            wrapper.render_containers = container_group.render_containers
            for func in container_group.bake_functions:
                bake_functions.append((wrapper.wobject, func))

    def call_bake_functions(self, camera, logical_size):
        """Call any collected bake functions."""
        # Enable pipelines to update data on the CPU. This usually includes
        # baking data into buffers. This is CPU intensive, but in practice
        # it is only used by a few materials.
        for wobject, func in self._bake_functions:
            func(wobject, camera, logical_size)

    def iter_compute_pipelines(self):
        """Generator that yields the collected compute pipelines."""
        for pipeline_container in self._compute_pipeline_containers:
            yield pipeline_container

    def iter_render_pipelines_per_pass_type(self):
        """Generator that yields (pass_type, wobjects), with pass_type 'opaque', 'transparency' or 'weighted'."""
        current_pass_type = ""
        current_pipeline_containers = []
        for wrapper in self._wobject_wrappers:
            if wrapper.pass_type != current_pass_type:
                if current_pipeline_containers:
                    yield (current_pass_type, current_pipeline_containers)
                current_pass_type = wrapper.pass_type
                current_pipeline_containers = []
            current_pipeline_containers.extend(wrapper.render_containers)
        if current_pipeline_containers:
            yield (current_pass_type, current_pipeline_containers)


class WgpuRenderer(RootEventHandler, Renderer):
    """Turns Scenes into rasterized images using wgpu.

    Parameters
    ----------
    target : WgpuCanvas or Texture
        The target to render to. It is also used to determine the size of the
        render buffer.
    pixel_scale : float, optional
        The scale between the internal resolution and the physical resolution of the canvas.
        Setting to None (default) selects 1 if the screens looks to be HiDPI and 2 otherwise.
    pixel_ratio : float, optional
        The ratio between the number of internal pixels versus the logical pixels on the canvas.
        If both ``pixel_ratio`` and ``pixel_scale`` are set, ``pixel_ratio`` is ignored.
    pixel_filter : str, PixelFilter, optional
        The type of interpolation / reconstruction filter to use. Default 'mitchell'.
    show_fps : bool
        Whether to display the frames per second. Beware that
        depending on the GUI toolkit, the canvas may impose a frame rate limit.
    sort_objects : bool
        If True, sort objects by depth before rendering. If False, the
        rendering order is mainly based on the objects ``render_order`` and position
        in the scene graph.
    enable_events : bool
        If True, forward wgpu events to pygfx's event system.
    gamma_correction : float
        The gamma correction to apply in the final render stage. Typically a
        number between 0.0 and 2.0. A value of 1.0 indicates no correction.
    ppaa : str, optional
        The post-processing anti-aliasing to apply: "default", "none", "fxaa", "ddaa".
        By default it resolves to "ddaa".
    """

    def __init__(
        self,
        target,
        *args,
        pixel_scale=None,
        pixel_ratio=None,
        pixel_filter: PixelFilter = "mitchell",
        show_fps=False,
        sort_objects=True,
        enable_events=True,
        gamma_correction=1.0,
        ppaa="default",
        **kwargs,
    ):
        blend_mode = kwargs.pop("blend_mode", None)
        super().__init__(*args, **kwargs)

        # blend_mode is deprecated; raise error with somewhat helpful message
        if blend_mode:
            self.blend_mode = blend_mode

        # Check and normalize inputs
        if not isinstance(target, (Texture, GfxTextureView, BaseRenderCanvas)):
            raise TypeError(
                f"Render target must be a Canvas or Texture, not a {target.__class__.__name__}"
            )
        self._target = target
        self.pixel_ratio = pixel_ratio
        if pixel_scale is not None:
            self.pixel_scale = pixel_scale

        # Make sure we have a shared object (the first renderer creates the instance)
        self._shared = get_shared()
        self._device = self._shared.device

        # Count number of objects encountered
        self._object_count = 0

        # Init counter to auto-clear
        self._renders_since_last_flush = 0

        # Cache renderstate objects for n draws
        self._renderstates_per_flush = []

        # Get target format
        self.gamma_correction = gamma_correction
        self._gamma_correction_srgb = 1.0
        if isinstance(target, BaseRenderCanvas):
            self._canvas_context = self._target.get_context("wgpu")
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
            # Also enable the texture for render and display usage
            self._target._wgpu_usage |= wgpu.TextureUsage.RENDER_ATTACHMENT
            self._target._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING

        self._blender = Blender()
        self._effect_passes = ()
        self.ppaa = ppaa
        self._name_of_texture_with_effects = (
            None  # none, or the blender's name of the texture
        )

        self.sort_objects = sort_objects

        # Prepare object that performs the final render step into a texture
        self._output_pass = OutputPass()
        self.pixel_filter = pixel_filter

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
    def pixel_scale(self) -> float:
        """The scale between the internal resolution and the physical resolution of the canvas.

        * If the scale is 1, the internal texture has the same size as the target.
        * If the scale is larger than 1, you're doing SSAA.
        * If the scale is smaller than 1, you're rendering at a low resolution, and then upscaling the result.

        Note that a ``pixel_scale`` of 1 or 2 is more performant than fractional values.

        Setting this value to ``None``, will select a hirez configuration: It
        selects 1 if the target looks like a HiDPI screen (i.e.
        ``canvas.pixel_ratio>=2``), and 2 otherwise. That way, the internal
        texture size is the same, regardless of the user's system/monitor.
        """
        return self._pixel_scale

    @pixel_scale.setter
    def pixel_scale(self, pixel_scale: None | int | float):
        if pixel_scale is None:
            # Select hirez config
            self._pixel_scale = 2.0  # default
            if isinstance(self._target, BaseRenderCanvas):
                target_pixel_ratio = self._target.get_pixel_ratio()
                if target_pixel_ratio >= 2.0:
                    self._pixel_scale = 1.0
        else:
            pixel_scale = float(pixel_scale)
            if pixel_scale < 0.1 or pixel_scale > 10:
                raise ValueError("renderer.pixel_scale must be bwteen 0.1 and 10.")
            self._pixel_scale = pixel_scale

    @property
    def pixel_ratio(self) -> float:
        """The ratio between the number of internal pixels versus the logical pixels on the canvas.

        ``pixel_ratio = pixel_scale * canvas.pixel_ratio``

        Setting this prop also changes the ``pixel_scale``. This can be used to
        configure the size of the internal texture relative to the canvas'
        *logical* size.

        Setting this value to ``None`` is the same as setting ``pixel_scale`` to None,
        and results in a ``pixel_ratio`` of at least 2.

        Note that setting ``pixel_ratio`` to 2.0 does not have the same effect, because the
        canvas pixel_ratio can be e.g. 1.5, in which case the resulting ``pixel_scale`` becomes fractional.
        """
        target_pixel_ratio = 1
        if isinstance(self._target, BaseRenderCanvas):
            target_pixel_ratio = self._target.get_pixel_ratio()
        return self._pixel_scale * target_pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, pixel_ratio: None | float):
        if pixel_ratio is None:
            self.pixel_scale = None
        else:
            target_pixel_ratio = 1
            if isinstance(self._target, BaseRenderCanvas):
                target_pixel_ratio = self._target.get_pixel_ratio()
            pixel_scale = pixel_ratio / target_pixel_ratio
            if 0.9 < pixel_scale < 1.1:
                pixel_scale = 1  # snap
            elif 1.9 < pixel_scale < 2.1:
                pixel_scale = 2  # snap
            self.pixel_scale = pixel_scale

    @property
    def pixel_filter(self) -> PixelFilter:
        """The type of interpolation / reconstruction filter to use when flushing the result.

        See :obj:`pygfx.utils.enums.PixelFilter`.

        The renderer renders everything to an internal texture, which,
        depending on the ``pixel_scale``, may have a different physical size than
        the target (i.e. canvas). In the process of rendering the result
        to the target, a filter is applied, resulting in SSAA if the
        target size is smaller, and upsampling when the target size is larger.
        When the internal texture has the same size as the target, no filter is applied (equivalent to nearest).

        The filter defines how the interpolation is done (when the source and target are not of the same size).
        """
        return self._output_pass.filter

    @pixel_filter.setter
    def pixel_filter(self, value: PixelFilter):
        # For backwards compatibility, allow 0, 1, 0.0, 1.0, False, and True.
        if value == 0:
            value = "nearest"
        elif value == 1:
            value = "mitchell"
        if not isinstance(value, str):
            raise TypeError("Pixel filter must be a str.")
        value = value.lower()
        if value not in PixelFilter.__args__:
            raise ValueError(
                f"Pixel filter must be one of {PixelFilter}, not {value!r}"
            )
        self._output_pass.filter = value
        self._pixel_filter = value

    @property
    def ppaa(
        self,
    ) -> Literal["default", "none", "fxaa", "ddaa"]:
        """The post-processing anti-aliasing to apply.

        * "default": use the value specified by ``PYGFX_DEFAULT_PPAA``, defaulting to "ddaa".
        * "none": do not apply aliasing.
        * "fxaa": applies Fast Approxomate AA, a common method.
        * "ddaa": applies Directional Diffusion AA, a modern improved method.

        The ``PYGFX_DEFAULT_PPAA`` environment variable can e.g. be set to "none" for image tests,
        so that the image tests don't fail when we update the ddaa method.

        Note that SSAA can be achieved by using a pixel_scale > 1. This can be well combined with PPAA,
        since the PPAA is applied before downsampling to the target texture.
        """
        ppaa = "none"
        for effect_pass in self._effect_passes:
            if isinstance(effect_pass, PPAAPass):
                ppaa = effect_pass.__class__.__name__.split("Pass")[0].lower()
        return ppaa

    @ppaa.setter
    def ppaa(self, ppaa: Literal["default", "none", "fxaa", "ddaa"]):
        if not isinstance(ppaa, str):
            raise TypeError(f"renderer.ppaa must be a string, not {ppaa!r}")

        # Collect list of effect passes that are not a ppaa
        effect_passes = []
        for effect_pass in self._effect_passes:
            if not isinstance(effect_pass, PPAAPass):
                effect_passes.append(effect_pass)

        # Handle default
        algorithm = ppaa.lower()
        if algorithm == "default":
            algorithm = os.getenv("PYGFX_DEFAULT_PPAA", "").lower()
            if not algorithm or algorithm == "default":
                algorithm = "ddaa"

        if algorithm == "none":
            pass  # don't add a pass
        elif algorithm == "ddaa":
            effect_passes.append(DDAAPass())
        elif algorithm == "fxaa":
            effect_passes.append(FXAAPass())
        else:
            raise ValueError(f"Invalid value for renderer.ppaa: {ppaa!r}")

        self.effect_passes = effect_passes

    @property
    def rect(self):
        """The rectangular viewport for the renderer area."""
        return (0, 0, *self.logical_size)

    @property
    def logical_size(self):
        """The size of the render target in logical pixels."""
        target = self._target
        if isinstance(target, BaseRenderCanvas):
            return target.get_logical_size()
        elif isinstance(target, Texture):
            return target.size[:2]  # assuming pixel-ratio 1
        else:
            raise TypeError(f"Unexpected render target {target.__class__.__name__}")

    @property
    def physical_size(self):
        """The physical size of the internal render texture."""
        target = self._target
        if isinstance(self._target, BaseRenderCanvas):
            target_physical_size = self._target.get_physical_size()
        else:
            target_physical_size = target.size[:2]
        w, h = target_physical_size
        pixel_scale = self._pixel_scale
        return max(1, int(w * pixel_scale)), max(1, int(h * pixel_scale))

    @property
    def blend_mode(self):
        raise DeprecationWarning(
            "renderer.blend_mode is removed. Use material.alpha_mode instead."
        )

    @blend_mode.setter
    def blend_mode(self, value):
        raise DeprecationWarning(
            "renderer.blend_mode is removed. Use material.alpha_mode instead."
        )

    @property
    def sort_objects(self):
        """Whether to sort world objects by depth before rendering. Default False.

        By default, the render order is defined by:

          1. the object's ``render_order`` property;
          2. whether the object is opaque/transparent/weighted/unknown;
          3. the object's distance to the camera;
          4. the object's position in the scene graph (based on a depth-first search).

        If ``sort_objects`` is ``False``, step 3 (sorting using the camera transform) is omitted.
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
        if isinstance(self._target, BaseRenderCanvas):
            self._target.request_draw()

    @property
    def effect_passes(self):
        """A tuple of ``EffectPass`` instances.

        Together they form a chain of post-processing passes to produce the final visual result.
        They are executed in order when the rendered image is flushed to the target (e.g. the screen).
        """
        return self._effect_passes

    @effect_passes.setter
    def effect_passes(self, effect_passes):
        effect_passes = tuple(effect_passes)
        for step in effect_passes:
            if not isinstance(step, EffectPass):
                raise TypeError(
                    f"A renderer effect-pass step must be an instance of EffectPass, not {step!r}"
                )
        self._effect_passes = effect_passes

    def clear(self, *, all=False, color=False, depth=False, weights=False):
        """Clear one or more of the render targets.

        Users typically don't need to use this method. But sometimes it can be convenient to e.g.
        render a scene, and then clear the depth before rendering another scene.

        * all: clear all render targets; a fully clean sheeth.
        * color: clear the color buffer to rgba all zeros.
        * depth: clear the depth buffer.
        * weights: clear the render targets for weighted blending (the accum and reveal textures).

        """
        if not (all or color or depth or weights):
            raise ValueError(
                "renderer.clear() needs at least all, color, depth, or weights set to True."
            )

        if all:
            self._blender.clear()
        else:
            if color:
                self._blender.texture_info["color"]["clear"] = True
            if depth:
                self._blender.texture_info["depth"]["clear"] = True
            if weights:
                self._blender.texture_info["accum"]["clear"] = True
                self._blender.texture_info["reveal"]["clear"] = True

    def render(
        self,
        scene: WorldObject,
        camera: Camera,
        *,
        rect=None,
        clear=None,
        flush=True,
        # deprecated:
        clear_color=None,
    ):
        """Render a scene with the specified camera as the viewpoint.

        Parameters:
            scene (WorldObject): The scene to render, a WorldObject that
                optionally has child objects.
            camera (Camera): The camera object to use, which defines the
                viewpoint and view transform.
            rect (tuple, optional): The rectangular region to draw into,
                expressed in logical pixels, a.k.a. the viewport.
            clear (bool, optional): Whether to clear the color and depth buffers
                before rendering. By default this is True on the first
                call to ``render()`` after a flush, and False otherwise.
            flush (bool, optional): Whether to flush the rendered result into
                the target (texture or canvas). Default True.
        """

        # A good time to reset this flag.
        self._name_of_texture_with_effects = None

        # Manage stored renderstate objects. Each renderstate object used will be stored at least a few draws.
        if self._renders_since_last_flush == 0:
            self._renderstates_per_flush.insert(0, [])
            self._renderstates_per_flush[16:] = []

        # Define whether to clear render targets
        if clear_color is not None:
            logger.warning(
                "renderer.render(.. clear_color) is deprecated in favor of 'clear'."
            )
        if clear is None:
            clear = self._renders_since_last_flush == 0
        if clear:
            self._blender.clear()

        self._renders_since_last_flush += 1

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
            physical_viewport = (*physical_viewport, 0, 1)
            scene_psize = physical_viewport[2], physical_viewport[3]
        else:
            raise ValueError(
                "The viewport rect must be None or 4 elements (x, y, w, h)."
            )

        # Apply the camera's native size (do this before we change scene_lsize based on view_offset)
        camera.set_view_size(*scene_lsize)

        # Camera view_offset overrides logical size
        ndc_offset = (1.0, 1.0, 0.0, 0.0)  # (ax ay bx by)  virtual_ndc = a * ndc + b
        if camera._view_offset is not None:
            scene_lsize = camera._view_offset["width"], camera._view_offset["height"]
            ndc_offset = camera._view_offset["ndc_offset"]

        # Allow objects to prepare just in time. When doing multiple
        # render calls, we don't want to spam. The clear flag is
        # a good indicator to detect the first render call.
        if clear:
            ev = WindowEvent(
                "before_render",
                target=None,
                root=self,
                width=logical_size[0],
                height=logical_size[1],
                pixel_ratio=pixel_ratio,
            )
            self.dispatch_event(ev)

        # Get a flat and sorted version of the scene.
        # This is also where wobject._update_object() is called
        view_matrix = None
        if self._sort_objects:
            view_matrix = camera.view_matrix  # == camera.world.inverse_matrix
        flat = FlatScene(scene, view_matrix, self._object_count)
        self._object_count = flat.object_count
        flat.sort()

        # Prepare the shared object
        self._shared.pre_render_hook()

        # Update stdinfo uniform buffer object that we'll use during this render call
        self._update_stdinfo_buffer(camera, scene_psize, scene_lsize, ndc_offset)

        # Get renderstate object
        renderstate = get_renderstate(flat.lights, self._blender)
        self._renderstates_per_flush[0].append(renderstate)

        # Make sure pipeline objects exist for all wobjects. This also collects the bake functons.
        flat.collect_pipelines_container_groups(renderstate)

        # Enable pipelines to update data on the CPU.
        flat.call_bake_functions(camera, logical_size)

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
            renderstate,
            flat,
            physical_viewport,
        )

        # Collect commands and submit
        self._device.queue.submit(command_buffers)

        if flush:
            self.flush()

    def flush(self, target=None):
        """Render the result into the target. This method is called
        automatically unless you use ``.render(..., flush=False)``.
        """

        # Print FPS
        now = time.perf_counter()
        if self._show_fps:
            if self._fps["count"] == 0:
                print(f"Time to first draw: {now - self._fps['start']:0.2f}")
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

        # Get the target texture view.
        if isinstance(target, BaseRenderCanvas):
            target_tex = self._canvas_context.get_current_texture().create_view()
        elif isinstance(target, Texture):
            need_mipmaps = target.generate_mipmaps
            target_tex = getattr(target, "_wgpu_default_view", None)
            if target_tex is None:
                target_tex = ensure_wgpu_object(GfxTextureView(target))
                target._wgpu_default_view = target_tex
        elif isinstance(target, GfxTextureView):
            need_mipmaps = target.texture.generate_mipmaps
            target_tex = ensure_wgpu_object(target)
        else:
            raise TypeError("Unexpected target type.")

        # Reset counter (so we can auto-clear the first next draw)
        self._renders_since_last_flush = 0

        # Start recording ...
        command_encoder = self._device.create_command_encoder()

        # Preparations
        src_name, dst_name = "color", "altcolor"
        src_usage = wgpu.TextureUsage.TEXTURE_BINDING
        dst_usage = wgpu.TextureUsage.RENDER_ATTACHMENT

        if self._name_of_texture_with_effects:
            # probably flushing a second time
            src_name = self._name_of_texture_with_effects
        else:
            # Apply any effect passes
            for step in self._effect_passes:
                color_tex = self._blender.get_texture_view(
                    src_name, src_usage, create_if_not_exist=True
                )
                dst_tex = self._blender.get_texture_view(
                    dst_name, dst_usage, create_if_not_exist=True
                )
                depth_tex = None
                if step.USES_DEPTH:
                    depth_tex = self._blender.get_texture_view(
                        "depth", src_usage, create_if_not_exist=False
                    )
                step.render(command_encoder, color_tex, depth_tex, dst_tex)
                # Pingpong
                src_name, dst_name = dst_name, src_name
            self._name_of_texture_with_effects = src_name

        # Apply copy-pass
        color_tex = self._blender.get_texture_view(src_name, src_usage)
        self._output_pass.gamma = self._gamma_correction * self._gamma_correction_srgb
        # self._output_pass.filter_strength = self._pixel_filter
        self._output_pass.render(command_encoder, color_tex, None, target_tex)

        self._device.queue.submit([command_encoder.finish()])

        if need_mipmaps:
            generate_texture_mipmaps(target)

    def _render_recording(
        self,
        renderstate,
        flat,
        physical_viewport,
    ):
        # You might think that this is slow for large number of world
        # object. But it is actually pretty good. It does iterate over
        # all world objects, and over stuff in each object. But that's
        # it, really.
        # todo: we may be able to speed this up with render bundles though

        command_encoder = self._device.create_command_encoder()

        # ----- compute pipelines

        compute_pass = command_encoder.begin_compute_pass()
        for compute_pipeline_container in flat.iter_compute_pipelines():
            compute_pipeline_container.dispatch(compute_pass)
        compute_pass.end()

        # ----- process shadow maps

        if flat.shadow_objects:
            lights = (
                flat.lights["point_lights"]
                + flat.lights["spot_lights"]
                + flat.lights["directional_lights"]
            )
            render_shadow_maps(lights, flat.shadow_objects, command_encoder)

        # ----- render in stages

        blender = self._blender
        rendered_something = False

        for iter in flat.iter_render_pipelines_per_pass_type():
            pass_type, render_pipeline_containers = iter

            # Only render this pass type if there are objects
            if not render_pipeline_containers:
                continue
            rendered_something = True

            render_pass = command_encoder.begin_render_pass(
                color_attachments=blender.get_color_attachments(pass_type),
                depth_stencil_attachment=blender.get_depth_attachment(),
                occlusion_query_set=None,
            )

            render_pass.set_viewport(*physical_viewport)

            for render_pipeline_container in render_pipeline_containers:
                render_pipeline_container.draw(render_pass, renderstate)

            render_pass.end()

            # Objects blended with weighted must be resolved into the color texture
            if pass_type == "weighted":
                self._blender.perform_weighted_resolve_pass(command_encoder)

        # Make sure the render targets exist (even if no objects are rendered)
        if not rendered_something:
            blender.get_color_attachments("normal")
            blender.get_depth_attachment()

        return [command_encoder.finish()]

    def _update_stdinfo_buffer(
        self, camera: Camera, physical_size, logical_size, ndc_offset
    ):
        # Update the stdinfo buffer's data
        # All matrices need to be transposed, because in WGSL they are column-major.
        # -> From the numpy p.o.v. the matrices are transposed, in wgsl they are
        #    upright again because the different interpretation of the memory.
        stdinfo_data = self._shared.uniform_buffer.data
        stdinfo_data["cam_transform"] = camera.world.inverse_matrix.T
        stdinfo_data["cam_transform_inv"] = camera.world.matrix.T
        stdinfo_data["projection_transform"] = camera.projection_matrix.T
        stdinfo_data["projection_transform_inv"] = camera.projection_matrix_inverse.T
        # stdinfo_data["ndc_to_world"].flat = la.mat_inverse(stdinfo_data["cam_transform"] @ stdinfo_data["projection_transform"])
        stdinfo_data["ndc_offset"] = ndc_offset
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

        color_tex = self._blender.get_texture("color")
        pick_tex = self._blender.get_texture("pick")

        # Sample
        encoder = self._device.create_command_encoder()
        if color_tex:
            self._copy_pixel(encoder, color_tex, float_pos, 0)
        if pick_tex:
            self._copy_pixel(encoder, pick_tex, float_pos, 8)
        self._device.queue.submit([encoder.finish()])

        # Collect data from the buffer
        self._pixel_info_buffer.map_sync("read")
        try:
            data = self._pixel_info_buffer.read_mapped()
        finally:
            self._pixel_info_buffer.unmap()
        color = Color(x / 255 for x in tuple(data[0:4].cast("B")))
        pick_value = tuple(data[8:16].cast("Q"))[0]  # noqa: RUF015
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
        w, h, _d = render_texture.size
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
        texture = self._blender.get_texture(
            self._name_of_texture_with_effects or "color"
        )
        size = texture.size
        if texture.format == "rgba16float":
            bytes_per_pixel = 8
            dtype = np.float16
        else:
            bytes_per_pixel = 4
            dtype = np.uint8

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

        return np.frombuffer(data, dtype).reshape(size[1], size[0], 4)

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
