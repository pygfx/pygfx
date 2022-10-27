"""
A global object shared by all renderers.
"""

import wgpu

from ...resources import Buffer
from ...utils.trackable import Trackable
from ...utils import array_from_shadertype


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


class Shared(Trackable):
    """An object to store global data to share between multiple wgpu
    renderers. Each renderer updates the data and then passes this down
    to the pipeline containers.

    The renderer instantiates and stores the singleton shared object.
    """

    def __init__(self, canvas):
        super().__init__()

        # Create adapter and device objects - there should be just one per canvas.
        # Having a global device provides the benefit that we can draw any object
        # anywhere.
        # We could pass the canvas to request_adapter(), so we get an adapter that is
        # at least compatible with the first canvas that a renderer is create for.
        # However, passing the object has been shown to prevent the creation of
        # a canvas (on Linux + wx), so, we never pass it for now.
        self._adapter = wgpu.request_adapter(
            canvas=None, power_preference="high-performance"
        )
        self._device = self.adapter.request_device(
            required_features=[], required_limits={}
        )

        # Create a uniform buffer for std info
        # Stored on _store so if we'd ever swap it out for another buffer,
        # the pipeline automatically update.
        self._store.uniform_buffer = Buffer(array_from_shadertype(stdinfo_uniform_type))
        self._store.uniform_buffer._wgpu_usage |= wgpu.BufferUsage.UNIFORM

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
    def glyph_atlas(self):
        """The shared glyph atlas (a texture view). TODO"""
        return self._store.glyph_atlas
        # todo: implement this as part of the text PR
