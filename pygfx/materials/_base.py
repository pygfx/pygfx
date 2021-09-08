from ..objects._base import ResourceContainer
from ..utils import array_from_shadertype
from ..resources import Buffer


class Material(ResourceContainer):
    """The base class for all materials.
    Materials define how an object is rendered, subject to certain properties.
    """

    uniform_type = dict(
        opacity=("float32",),
    )

    def __init__(self, *, opacity=1):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self.opacity = opacity

    # def _set_property_values(self, **kwargs):
    #     for key, val in kwargs.items():
    #         if isinstance(getattr(type(self), key, None) property):
    #             setattr(self, key, val)
    #         else:
    #             raise KeyError(f"{self.__class__.__name__} does not have property '{key}'.")

    def _wgpu_get_pick_info(self, pick_value):
        """Given a 4 element tuple, sampled from the pick texture,
        return info about what was picked in the object. The first value
        refers to the object id. The use of the remaining values differs
        per material.
        """
        # Note that this is a private friend-method of the renderer.
        return {}

    @property
    def opacity(self):
        """The opacity (a.k.a. alpha value) applied to this material, expressed
        as a value between 0 and 1. If the material contains any
        non-opaque fragments, these are simply scaled by this value.
        """
        return float(self.uniform_buffer.data["opacity"])

    @opacity.setter
    def opacity(self, value):
        self.uniform_buffer.data["opacity"] = min(max(float(value), 0), 1)
        self.uniform_buffer.update_range(0, 1)
