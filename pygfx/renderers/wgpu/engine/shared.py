"""
A global object shared by all renderers.
"""

import wgpu

from ....resources import Resource, Buffer
from ....utils.trackable import Trackable
from ....utils import array_from_shadertype
from ....utils.text import glyph_atlas

from .utils import gpu_caches


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
    # textures, which is sad for e.g. volume rendering. We require the
    # 'float32-filterable' feature for this.
    #
    # Previously (when this feature was not yet available) we used the
    # native-only feature 'texture_adapter_specific_format_features',
    # but that one enables far more than we want.
    #
    # Technically we can check if the feature is available and fall
    # back (in some way) when it's not available, but this easily
    # results in complex code. I think we can probably get away with
    # requiring only a few features that are available on the main
    # target platforms.

    _features = set(["float32-filterable"])

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

        self._create_diagnostics()

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

    def _create_diagnostics(self):
        # Since diagnostics objects, are shown in order of creation,
        # its good to instantiate our diagnostics objects right after loading
        # the wgpu_native backend from wgpu.
        PyGfxAdapterInfoDiagnostics("pygfx_adapter_info")
        PyGfxFeaturesDiagnostics("pygfx_features")
        PyGfxLimitsDiagnostics("pygfx_limits")
        PygfxCacheDiagnostics("pygfx_caches")
        PygfxResourceDiagnostics("pygfx_resources")


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


class PyGfxAdapterInfoDiagnostics(wgpu.DiagnosticsBase):

    def get_dict(self):
        shared = get_shared()
        adapter = shared.adapter
        return adapter.request_adapter_info()


class PyGfxFeaturesDiagnostics(wgpu.DiagnosticsBase):

    def get_dict(self):
        shared = get_shared()
        adapter = shared.adapter
        device = shared.device

        feature_names = list(wgpu.FeatureName)
        feature_names += sorted(adapter.features.difference(wgpu.FeatureName))
        result = {}
        for key in feature_names:
            result[key] = {
                "adapter": key in adapter.features,
                "device": key in device.features,
            }
        return result


class PyGfxLimitsDiagnostics(wgpu.DiagnosticsBase):

    def get_dict(self):
        shared = get_shared()
        adapter = shared.adapter
        device = shared.device

        result = {}
        for key in adapter.limits.keys():
            result[key] = {
                "adapter": adapter.limits[key],
                "device": device.limits.get(key, False),
            }
        return result


class PygfxCacheDiagnostics(wgpu.DiagnosticsBase):

    def get_dict(self):
        result = {}
        for cache_name, stats in gpu_caches.get_stats().items():
            count, hits, misses = stats
            result[cache_name] = {"count": count, "hits": hits, "misses": misses}
        return result


class PygfxResourceDiagnostics(wgpu.DiagnosticsBase):

    def get_dict(self):
        return Resource._resource_counts


def print_wgpu_report():
    """Print a report on the internal status of WGPU. Can be useful
    in debugging, and for providing details when making a bug report.
    """
    wgpu.diagnostics.print_report()