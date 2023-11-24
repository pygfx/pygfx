"""
A global object shared by all renderers.
"""

import wgpu

from ...resources import Resource, Buffer
from ...utils.trackable import Trackable
from ...utils import array_from_shadertype
from ...utils.text import glyph_atlas
from ._utils import gpu_caches


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
)


class Shared(Trackable):
    """An object to store global data to share between multiple wgpu
    renderers. Each renderer updates the data and then passes this down
    to the pipeline containers.

    The renderer instantiates and stores the singleton shared object.
    """

    # Vanilla WGPU does not support interpolating samplers for float32
    # textures, which is sad for e.g. volume rendering. WebGPU specifies
    # the 'float32-filterable' feature for this, but its not yet available
    # in wgpu-core. Fortunately, we can enable the same functionality via
    # the native-only feature 'texture_adapter_specific_format_features'.

    _features = set(["texture_adapter_specific_format_features"])

    _instance = None

    def __init__(self, *, canvas=None):
        super().__init__()

        assert Shared._instance is None
        Shared._instance = self

        # Create adapter and device objects - there should be just one per
        # process. Having a global device provides the benefit that we can draw
        # any object anywhere.
        self._adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self._device = self.adapter.request_device(
            required_features=list(Shared._features), required_limits={}
        )

        # Create a uniform buffer for std info
        # Stored on _store so if we'd ever swap it out for another buffer,
        # the pipeline automatically update.
        self._store.uniform_buffer = Buffer(array_from_shadertype(stdinfo_uniform_type))
        self._store.uniform_buffer._wgpu_usage |= wgpu.BufferUsage.UNIFORM

        # Init glyph atlas texture
        self._store.glyph_atlas_texture = None
        self._store.glyph_atlas_info_buffer = None
        self.pre_render_hook()

    def pre_render_hook(self):
        """Called by the renderer on the beginning of a draw."""
        tex = glyph_atlas.texture
        if tex is not self._store["glyph_atlas_texture"]:
            self._store.glyph_atlas_texture = tex
        buffer = glyph_atlas.info_buffer
        if buffer is not self._store["glyph_atlas_info_buffer"]:
            self._store.glyph_atlas_info_buffer = buffer

    @classmethod
    def get_instance(cls):
        return cls._instance

    @property
    def adapter(self):
        """The shared WGPU adapter object."""
        return self._adapter

    @property
    def device(self):
        """The shared WGPU device object."""
        return self._device

    @property
    def uniform_buffer(self):
        """The shared uniform buffer in which the renderer puts
        information about the canvas and camera.
        """
        return self._store.uniform_buffer

    @property
    def glyph_atlas_texture(self):
        """The shared glyph atlas (as a texture view and sampler) for objects that want to render text."""
        return self._store.glyph_atlas_texture

    @property
    def glyph_atlas_info_buffer(self):
        """A buffer containing per-glyph metadata (rects and more)."""
        return self._store.glyph_atlas_info_buffer


def enable_wgpu_features(*features):
    """Enable specific features (as strings) on the wgpu device.

    WARNING: enabling features means that your code may not work on all
    devices. The point of wgpu is that it can make a promise that a
    visualization works and looks the same on any device. Using features
    breaks that promise, and may cause your code to not work on e.g.
    mobile devices or certain operating systems.

    This function must be called before the first wgpu device is created.
    In practice this means before the first ``Renderer`` is created.
    It can be called multiple times to enable more features.

    For more information on features:

    * ``wgpu.FeatureName`` for all possible official features.
    * ``renderer.device.adapter.features`` for the features available on the current system.
    * ``renderer.device.features`` for the currently enabled features.
    * https://gpuweb.github.io/gpuweb/#gpufeaturename for the official webgpu features (excl. native features).
    * https://docs.rs/wgpu/latest/wgpu/struct.Features.html for the features and their limitations in wgpu-core.
    """
    if Shared._instance is not None:
        raise RuntimeError(
            "The enable_wgpu_features() function must be called before creating the first renderer."
        )
    Shared._features.update(features)


def get_shared():
    """Get the globally shared instance. Creates it if it does not yet exist.
    This should not be called at the import time of any module.
    Use this to get the global device: `get_shared().device`.
    """
    if Shared._instance is None:
        Shared()
    return Shared._instance


def print_wgpu_report():
    """Print a report on the internal status of WGPU. Can be useful
    in debugging, and for providing details when making a bug report.
    """
    shared = get_shared()
    adapter = shared.adapter
    device = shared.device

    print()
    print("ADAPTER INFO:")
    for key, val in adapter.request_adapter_info().items():
        print(f"{key.rjust(50)}: {val}")

    print()
    print("FEATURES:".ljust(50), "adapter".rjust(10), "device".rjust(10))
    feature_names = list(wgpu.FeatureName)
    feature_names += sorted(adapter.features.difference(wgpu.FeatureName))
    for key in feature_names:
        adapter_has_it = "Y" if key in adapter.features else "-"
        device_has_it = "Y" if key in device.features else "-"
        print(f"{key}:".rjust(50), adapter_has_it.rjust(10), device_has_it.rjust(10))

    print()
    print("LIMITS:".ljust(50), "adapter".rjust(10), "device".rjust(10))
    for key in adapter.limits.keys():
        val1 = adapter.limits[key]
        val2 = device.limits.get(key, "-")
        print(f"{key}:".rjust(50), str(val1).rjust(10), str(val2).rjust(10))

    print()
    print("CACHES:".ljust(20), "Count".rjust(10), "Hits".rjust(10), "Misses".rjust(10))
    for cache_name, stats in gpu_caches.get_stats().items():
        count, hits, misses = stats
        print(
            f"{cache_name}".rjust(20),
            str(count).rjust(10),
            str(hits).rjust(10),
            str(misses).rjust(10),
        )

    print()
    print("RESOURCES:".ljust(20), "Count".rjust(10))
    for name, count in Resource._resource_counts.items():
        print(
            f"{name}:".rjust(20),
            str(count).rjust(10),
        )

    print()
    wgpu.print_report()
