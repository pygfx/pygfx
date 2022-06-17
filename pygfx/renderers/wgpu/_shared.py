"""
A global object shared by all renderers.
"""

import wgpu

from ...resources import Buffer
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


class Shared:
    """An object to store global data to share between multiple wgpu
    renderers. Each renderer updates the data and then passes this down
    to the pipeline containers.

    The renderer instantiates and stores the singleton shared object.
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
        self.uniform_buffer = Buffer(array_from_shadertype(stdinfo_uniform_type))
        self.uniform_buffer._wgpu_usage |= wgpu.BufferUsage.UNIFORM

