"""
Experimental support for compute shaders.
This should eventually be a more generic class, and move to wgpu.utils, or a new library.
See https://github.com/pygfx/wgpu-py/issues/704
"""

import time

from typing import Optional, Union
import wgpu
import pygfx as gfx


# TODO: move this into wgpu/pygfx/new lib
# TODO: ability to concatenate multiple steps
# TODO: add support for uniforms
class ComputeShader:
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
        and in debugging messages. If not set, use the entry_point.
    report_time : bool
        When set to True, will print the spent time to run the shader.
    """

    def __init__(
        self,
        wgsl,
        *,
        entry_point: Optional[str] = None,
        label: Optional[str] = None,
        report_time: bool = False,
    ):
        # Fixed
        self._wgsl = wgsl
        self._entry_point = entry_point
        self._label = label or entry_point or ""
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
            raise TypeError(f"ComputeShader resource index must be int, not {index!r}.")
        if not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture)
        ):
            raise TypeError(
                f"ComputeShader resource value must be gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, or wgpu.GPUTexture, not {resource!r}"
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
            raise TypeError(f"ComputeShader constant name must be str, not {name!r}.")
        if not (value is None or isinstance(value, (bool, int, float))):
            raise TypeError(
                f"ComputeShader constant value must be bool, int, float, or None, not {value!r}."
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
