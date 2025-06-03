"""
This module implements the PipelineContainer (for compute and render pipelines).
This object is responsible for creating the native wgpu objects and doing the
actual dispatching / drawing.
"""

import sys
import wgpu

import os

from ....utils import logger
from ....utils.weak import WeakAssociativeContainer
from ....utils.trackable import PropTracker

from ..shader.base import ShaderInterface
from .utils import registry, GpuCache, hash_from_value
from .shared import get_shared
from .binding import Binding


PIPELINE_CONTAINER_GROUPS = WeakAssociativeContainer()

# These caches use a WeakValueDictionary; they don't actually store object, but
# enable sharing gpu resources for similar objects. It makes creating such
# objects faster (i.e. faster startup). It also saves gpu resources. It does not
# necessarily make the visualization faster.
LAYOUT_CACHE = GpuCache("layouts")
BINDING_CACHE = GpuCache("bindings")
SHADER_CACHE = GpuCache("shader_modules")
PIPELINE_CACHE = GpuCache("pipelines")

PRINT_WGSL_ON_ERROR = os.environ.get(
    "PYGFX_PRINT_WGSL_ON_COMPILATION_ERROR", "0"
).lower() not in ["false", "0"]


def get_cached_bind_group_layout(device, *args):
    key = "bind_group_layout", hash_from_value(args)
    result = LAYOUT_CACHE.get(key)
    if result is None:
        (entries,) = args
        result = device.create_bind_group_layout(entries=entries)

        LAYOUT_CACHE.set(key, result)

    return result


def get_cached_pipeline_layout(device, *args):
    key = "pipeline_layout", hash_from_value(args)
    result = LAYOUT_CACHE.get(key)
    if result is None:
        (bind_group_layouts,) = args
        result = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)

        LAYOUT_CACHE.set(key, result)

    return result


def get_cached_bind_group(device, *args):
    key = "bind_group", hash_from_value(args)
    result = BINDING_CACHE.get(key)
    if result is None:
        layout, entries = args
        result = device.create_bind_group(layout=layout, entries=entries)

        BINDING_CACHE.set(key, result)

    return result


def get_cached_shader_module(device, shader, shader_kwargs):
    # Using a key that *defines* the wgsl - rather than the wgsl itself
    # - avoids the templating to be applied on a cache hit, which saves
    # considerable time!
    key = "shader", shader.hash, hash_from_value(shader_kwargs)

    result = SHADER_CACHE.get(key)
    if result is None:
        wgsl = shader.generate_wgsl(**shader_kwargs)
        try:
            result = device.create_shader_module(code=wgsl)
        except wgpu.GPUValidationError:
            # No need to be super fancy in the formatting but we want to
            # help the users find their bugs
            # We have seen some shaders with close to 1000 lines of code
            # So we print numbers that would be aligned up to 5 digits.
            # Since this error may be confusing for end users due to the sheer
            # volume of text printed, developers must enable this with
            # PYGFX_PRINT_WGSL_ON_COMPILATION_ERROR
            if PRINT_WGSL_ON_ERROR:
                wgsl_with_line_numbers = "\n".join(
                    f"{i + 1:5d}: {line}" for i, line in enumerate(wgsl.splitlines())
                )
                print(wgsl_with_line_numbers, file=sys.stderr)
            raise

        SHADER_CACHE.set(key, result)

    return result


def get_cached_compute_pipeline(device, *args):
    key = "compute_pipeline", hash_from_value(args)
    result = PIPELINE_CACHE.get(key)
    if result is None:
        pipeline_layout, shader_module = args

        result = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )

        PIPELINE_CACHE.set(key, result)

    return result


def get_cached_render_pipeline(device, *args):
    key = "render_pipeline", hash_from_value(args)
    result = PIPELINE_CACHE.get(key)
    if result is None:
        (
            pipeline_layout,
            shader_module,
            primitive_topology,
            strip_index_format,
            cull_mode,
            depth_descriptor,
            color_descriptors,
        ) = args

        result = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [],
            },
            primitive={
                "topology": primitive_topology,
                "strip_index_format": strip_index_format,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": cull_mode,
            },
            depth_stencil=depth_descriptor,
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": color_descriptors,
            },
        )

        PIPELINE_CACHE.set(key, result)

    return result


def get_pipeline_container_group(wobject, renderstate):
    """Return the PipelineContainerGroup object for wobject.

    This is the entrypoint for the renderer.

    The returned object has attributes ``bake_functions``, ``compute_containers``, and ``render_containers``.
    The latter two are lists to PipelineContainer objects. Most wobjects have just a single pipeline,
    but some may do preprocessing in a compute shader and/or consist of multiple passes.

    This call updates the GPU objects associated with the given wobject. It returns
    quickly if no changes are needed.
    """

    # Get pipeline container group. They are associated weakly by wobject and
    # renderstate. By also associating with the material the material can be
    # used by users to 'store' the pipeline, and hot-swap it. When any of these
    # objects (wobject, renderstate, material) is removed by the gc, the
    # associated pipeline_container object is removed as well.
    pcg_key = wobject, renderstate, wobject.material

    pipeline_container_group = PIPELINE_CONTAINER_GROUPS.get(pcg_key, None)
    if pipeline_container_group is None:
        pipeline_container_group = PipelineContainerGroup()
        PIPELINE_CONTAINER_GROUPS[pcg_key] = pipeline_container_group

    # Update. Quickly returns if no work to do.
    pipeline_container_group.update(wobject, renderstate)

    # Return the pipeline container group
    return pipeline_container_group


class PipelineContainerGroup:
    """Pipeline countainer group.

    This is a thin wrapper for a list of compute pipeline containers and render pipeline containers.
    The purpose of this object is to obtain the appropriate shader objects and store them.
    """

    def __init__(self):
        self.tracker = PropTracker()
        self.initialized = False
        # Public attributes used by the renderer
        self.compute_containers = None
        self.render_containers = None
        self.bake_functions = None

    def update(self, wobject, renderstate):
        """Update the pipeline containers. Creates (and re-creates) the containers if necessary."""

        # Get what has changed
        tracker = self.tracker
        changed = tracker.pop_changed()

        if not self.initialized:
            self.initialized = True
            changed.add("create")

        if not changed:
            return

        if "create" in changed:
            self.compute_containers = ()
            self.render_containers = ()
            self.bake_functions = ()

            # Get render function for this world object,
            # and use it to get a high-level description of pipelines.
            renderfunc = registry.get_render_function(wobject)
            if renderfunc is None:
                raise ValueError(
                    f"Could not get a render function for {wobject.__class__.__name__} "
                    f"with {wobject.material.__class__.__name__}"
                )

            # Call render function
            with tracker.track_usage("create"):
                shaders = renderfunc(wobject)
                if isinstance(shaders, ShaderInterface):
                    shaders = [shaders]

            # Divide result over two bins, one for compute, and one for render. Plus collect bake funcs.
            compute_containers = []
            render_containers = []
            bake_functions = []
            for shader in shaders:
                assert isinstance(shader, ShaderInterface)
                if shader.type == "compute":
                    compute_containers.append(ComputePipelineContainer(shader))
                elif shader.type == "render":
                    render_containers.append(RenderPipelineContainer(shader))
                else:
                    raise ValueError(f"Shader type {shader.type} is unknown.")
                if shader.needs_bake_function:
                    bake_functions.append(shader.bake_function)

            # Store results
            self.compute_containers = tuple(compute_containers)
            self.render_containers = tuple(render_containers)
            self.bake_functions = tuple(bake_functions)

        # If something has changed, update containers
        if changed:
            for container in self.compute_containers:
                container.update(wobject, renderstate, changed, tracker)
            for container in self.render_containers:
                container.update(wobject, renderstate, changed, tracker)


class PipelineContainer:
    """Object that wraps a set of wgpu pipeline objects for a single Shader object.

    A PipelineContainer is created for a specific combination of WorldObject and Renderstate
    (each differtent combination results in a different PipelineContainer).

    One shader results into multiple pipelines because of different render passes.

    The intermediate steps are also stored. When a dependency of a certain step
    changes (which we track) then only the steps below it need to be re-run.
    """

    def __init__(self, shader: ShaderInterface):
        self.shader = shader  # the corresponding ShaderInterface object
        self.shared = get_shared()  # the globally Shared object
        self.device = self.shared.device  # the global device

        # Dict to store info on the wobject that affects shaders or pipeline.
        # Fields are set in a tracking-context to make sure things update accordingly.
        self.wobject_info = {}

        # The info that the shader generates
        self.shader_hash = b""
        self.bindings_dicts = None  # dict of dict of bindings
        self.pipeline_info = None  # dict
        self.render_info = None  # dict

        # The wgpu objects that we generate. These are dicts; keys are pass indices.
        # For compute shaders the only key is 0.
        self.wgpu_shaders = {}
        self.wgpu_pipelines = {}

        # The bindings map group_index -> group
        self.bind_group_layout_descriptors = []
        self.wgpu_bind_group_layouts = []
        self.wgpu_bind_groups = []

        # A flag to indicate that an error occurred and we cannot dispatch
        self.broken = False

    def _check_bindings(self):
        assert isinstance(self.bindings_dicts, dict), "bindings_dicts must be a dict"
        for key1, bindings_dict in self.bindings_dicts.items():
            assert isinstance(key1, int), f"bindings slot must be int, not {key1}"
            assert isinstance(bindings_dict, dict), "bindings_dict must be a dict"
            for key2, b in bindings_dict.items():
                assert isinstance(key2, int), f"bind group slot must be int, not {key2}"
                assert isinstance(b, Binding), f"binding must be Binding, not {b}"

    def update(self, wobject, renderstate, changed, tracker):
        """Make sure that the pipeline is up-to-date for the given renderstate."""

        # Ensure that the information provided by the shader is up-to-date
        if changed:
            try:
                self.update_shader_data(wobject, changed, tracker)
            except Exception as err:
                self.broken = 1
                raise err
            else:
                self.broken = 0

        # Ensure that the (renderstate specific) wgpu objects are up-to-date
        if not self.broken:
            try:
                self.update_wgpu_data(renderstate, changed)
            except Exception as err:
                self.broken = 2
                raise err
            else:
                self.broken = False

        if changed:
            logger.info(f"{wobject} shader update: {', '.join(sorted(changed))}.")

    def update_shader_data(self, wobject, changed, tracker):
        """Update the info that applies to all passes and renderstates."""

        if "create" in changed or "reset" in changed:
            with tracker.track_usage("reset"):
                self.wobject_info["pick_write"] = wobject.material.pick_write
            changed.update(("bindings", "pipeline_info", "render_info"))
            self.wgpu_shaders = {}

        if "bindings" in changed:
            with tracker.track_usage("!bindings"):
                self.bindings_dicts = self.shader.get_bindings_info(
                    wobject, self.shared
                )
            self._check_bindings()
            self.update_shader_hash()
            self.update_bind_groups()

        if "pipeline_info" in changed:
            with tracker.track_usage("pipeline_info"):
                self.pipeline_info = self.shader.get_pipeline_info(wobject, self.shared)
                self.wobject_info["depth_test"] = wobject.material.depth_test
            self._check_pipeline_info()
            changed.add("render_info")
            self.wgpu_pipelines = {}

        if "render_info" in changed:
            with tracker.track_usage("render_info"):
                self.render_info = self.shader.get_render_info(wobject, self.shared)
            self._check_render_info()

    def update_wgpu_data(self, renderstate, changed):
        """Update the actual wgpu objects."""

        # Note: from here-on, we cannot access the wobject anymore, because any tracking should
        # be done in update_shader_data(). If you find that info on the wobject is needed,
        # add that info to self.wobject_info under the appropriate tracking context.

        # Determine what render-passes apply, for this combination of shader and blender
        if isinstance(self, RenderPipelineContainer):
            render_mask = self.render_info["render_mask"]
            blender = renderstate.blender
            pass_indices = []
            for pass_index in range(blender.get_pass_count()):
                if not render_mask & blender.passes[pass_index].render_mask:
                    continue
                if not blender.get_color_descriptors(
                    pass_index, self.wobject_info["pick_write"]
                ):
                    continue
                pass_indices.append(pass_index)
        else:
            pass_indices = [0]

        # Update shaders
        for pass_index in pass_indices:
            if pass_index not in self.wgpu_shaders:
                changed.add("shader")
                self.wgpu_shaders[pass_index] = self._get_shader(
                    pass_index, renderstate
                )
                self.wgpu_pipelines.pop(pass_index, None)

        # Update pipelines
        for pass_index in pass_indices:
            if pass_index not in self.wgpu_pipelines:
                changed.add("pipeline")
                self.wgpu_pipelines[pass_index] = self._compose_pipeline(
                    pass_index, renderstate
                )

    def update_shader_hash(self):
        """Update the shader hash, invalidating the wgpu shaders if it changed."""
        sh = self.shader.hash
        if sh != self.shader_hash:
            self.shader_hash = sh
            self.wgpu_shaders = {}

    def update_bind_groups(self):
        """
        - Calculate new bind_group_layout_descriptors (simple dicts).
        - When this has changed from our last version, we also update
          wgpu_bind_group_layouts and reset self.wgpu_pipelines
        - Calculate new wgpu_bind_groups.
        """

        # Check the bindings structure
        for i, bindings_dict in self.bindings_dicts.items():
            assert isinstance(i, int)
            assert isinstance(bindings_dict, dict)
            for j, b in bindings_dict.items():
                assert isinstance(j, int)
                assert isinstance(b, Binding)

        # Create two new dicts that correspond closely to the bindings.
        # Except this turns the dicts into lists.
        # These are the descriptors to create the wgpu bind groups and
        # bind group layouts.
        bg_descriptors = []
        bg_layout_descriptors = []
        for group_id in sorted(self.bindings_dicts.keys()):
            bindings_dict = self.bindings_dicts[group_id]
            while len(bg_descriptors) <= group_id:
                bg_descriptors.append([])
                bg_layout_descriptors.append([])
            bg_descriptor = bg_descriptors[group_id]
            bg_layout_descriptor = bg_layout_descriptors[group_id]
            for slot in sorted(bindings_dict.keys()):
                binding = bindings_dict[slot]
                binding_des, binding_layout_des = binding.get_bind_group_descriptors(
                    slot
                )
                bg_descriptor.append(binding_des)
                bg_layout_descriptor.append(binding_layout_des)

        # Clean up any trailing empty descriptors
        while bg_descriptors and not bg_descriptors[-1]:
            bg_descriptors.pop(-1)
            bg_layout_descriptors.pop(-1)

        # If the layout has changed, we need a new pipeline
        if self.bind_group_layout_descriptors != bg_layout_descriptors:
            self.bind_group_layout_descriptors = bg_layout_descriptors
            # Invalidate the pipeline
            self.wgpu_pipelines = {}
            # Create wgpu objects for the bind group layouts
            self.wgpu_bind_group_layouts = []
            for bg_layout_descriptor in bg_layout_descriptors:
                bind_group_layout = get_cached_bind_group_layout(
                    self.device, bg_layout_descriptor
                )
                self.wgpu_bind_group_layouts.append(bind_group_layout)

        # Always create wgpu objects for the bind groups. Note that this dict
        # includes the buffer texture objects.
        self.wgpu_bind_groups = []
        for bg_descriptor, layout in zip(
            bg_descriptors, self.wgpu_bind_group_layouts, strict=True
        ):
            bind_group = get_cached_bind_group(self.device, layout, bg_descriptor)
            self.wgpu_bind_groups.append(bind_group)


class ComputePipelineContainer(PipelineContainer):
    """Container for compute pipelines."""

    # These checks are here to validate the output of the shader
    # methods, not for user code. So its ok to use assertions here.

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        assert set(pipeline_info.keys()) == set(), f"{pipeline_info.keys()}"

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        assert set(render_info.keys()) == {"indices"}, f"{render_info.keys()}"

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) == 3

    def _get_shader(self, pass_index, renderstate):
        """Get the wgpu shader module for the templated wgsl shade."""
        shader_kwargs = {}
        return get_cached_shader_module(self.device, self.shader, shader_kwargs)

    def _compose_pipeline(self, pass_index, renderstate):
        """Create the wgpu pipeline object from the shader and bind group layouts."""

        # Create pipeline layout object from list of layouts
        pipeline_layout = get_cached_pipeline_layout(
            self.device, self.wgpu_bind_group_layouts
        )

        # Create pipeline object
        return get_cached_compute_pipeline(
            self.device, pipeline_layout, self.wgpu_shaders[0]
        )

    def dispatch(self, compute_pass):
        """Dispatch the pipeline, doing the actual compute job."""
        if self.broken:
            return

        # Collect what's needed
        pipeline = self.wgpu_pipelines[0]
        indices = self.render_info["indices"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and bindings
        compute_pass.set_pipeline(pipeline)
        for bind_group_id, bind_group in enumerate(bind_groups):
            compute_pass.set_bind_group(bind_group_id, bind_group, [], 0, 999999)

        # Compute!
        compute_pass.dispatch_workgroups(*indices)


class RenderPipelineContainer(PipelineContainer):
    """Container for render pipelines."""

    def __init__(self, shader):
        super().__init__(shader)
        self.strip_index_format = 0

    def _check_pipeline_info(self):
        pipeline_info = self.pipeline_info
        assert isinstance(pipeline_info, dict)

        expected = {"cull_mode", "primitive_topology"}
        assert set(pipeline_info.keys()) == expected, f"{pipeline_info.keys()}"
        self.update_strip_index_format()

    def _check_render_info(self):
        render_info = self.render_info
        assert isinstance(render_info, dict)

        expected = {"indices", "render_mask"}
        assert set(render_info.keys()) == expected, f"{render_info.keys()}"

        indices = render_info["indices"]
        assert isinstance(indices, (tuple, list))
        assert all(isinstance(i, int) for i in indices)
        assert len(indices) in (2, 4, 5)
        if len(indices) == 2:
            render_info["indices"] = indices[0], indices[1], 0, 0

        render_mask = render_info["render_mask"]
        assert isinstance(render_mask, int) and render_mask in (1, 2, 3)

    def update_strip_index_format(self):
        if not self.bindings_dicts or not self.pipeline_info:
            return
        # Set strip_index_format
        index_format = wgpu.IndexFormat.uint32
        strip_index_format = 0
        if "strip" in self.pipeline_info["primitive_topology"]:
            strip_index_format = index_format
        # Trigger a pipeline rebuild?
        if self.strip_index_format != strip_index_format:
            self.strip_index_format = strip_index_format
            self.wgpu_pipelines = {}

    def _get_shader(self, pass_index, renderstate):
        """Get the wgpu shader module for the templated shader."""
        blender = renderstate.blender
        renderstate_bind_group_index = len(self.wgpu_bind_groups)

        blender_kwargs = blender.get_shader_kwargs(pass_index)
        renderstate_kwargs = renderstate.get_shader_kwargs(renderstate_bind_group_index)
        shader_kwargs = blender_kwargs.copy()
        shader_kwargs.update(renderstate_kwargs)
        shader_kwargs["write_pick"] &= self.wobject_info["pick_write"]

        return get_cached_shader_module(self.device, self.shader, shader_kwargs)

    def _compose_pipeline(self, pass_index, renderstate):
        """Create the wgpu pipeline object from the shader, bind group layouts and other pipeline info."""

        strip_index_format = self.strip_index_format
        primitive_topology = self.pipeline_info["primitive_topology"]
        cull_mode = self.pipeline_info["cull_mode"]

        # Create pipeline layout object from list of layouts
        renderstate_bind_group_layout = renderstate.wgpu_bind_group_layout
        bind_group_layouts = [
            *self.wgpu_bind_group_layouts,
            renderstate_bind_group_layout,
        ]

        pipeline_layout = get_cached_pipeline_layout(self.device, bind_group_layouts)

        # Instantiate the pipeline objects.
        # Note: The pipeline relies on the color and depth descriptors, which
        # include the texture format and a few other static things.
        # This step should *not* rerun when e.g. the canvas resizes.
        blender = renderstate.blender
        depth_test = self.wobject_info["depth_test"]
        color_descriptors = blender.get_color_descriptors(
            pass_index, self.wobject_info["pick_write"]
        )
        depth_descriptor = blender.get_depth_descriptor(pass_index, depth_test)
        shader_module = self.wgpu_shaders[pass_index]

        return get_cached_render_pipeline(
            self.device,
            pipeline_layout,
            shader_module,
            primitive_topology,
            strip_index_format,
            cull_mode,
            depth_descriptor,
            color_descriptors,
        )

    def draw(self, render_pass, renderstate, pass_index, render_mask):
        """Draw the pipeline, doing the actual rendering job."""
        if self.broken:
            return

        if not (render_mask & self.render_info["render_mask"]):
            return

        # Collect what's needed
        pipeline = self.wgpu_pipelines[pass_index]
        indices = self.render_info["indices"]
        bind_groups = self.wgpu_bind_groups

        # Set pipeline and bindings
        render_pass.set_pipeline(pipeline)

        for bind_group_id, bind_group in enumerate(bind_groups):
            render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)

        renderstate_bind_group_id = len(bind_groups)
        renderstate_bind_group = renderstate.wgpu_bind_group
        render_pass.set_bind_group(
            renderstate_bind_group_id, renderstate_bind_group, [], 0, 99
        )

        # Draw!
        # draw(count_vertex, count_instance, first_vertex, first_instance)
        render_pass.draw(*indices)
