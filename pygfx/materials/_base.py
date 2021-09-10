from ..objects._base import ResourceContainer
from ..utils import array_from_shadertype
from ..resources import Buffer


class Material(ResourceContainer):
    """The base class for all materials.
    Materials define how an object is rendered, subject to certain properties.
    """

    uniform_type = dict(
        opacity=("float32",),
        clipping_planes=("float32", 4),  # todo: must support array of vec4's
    )

    def __init__(self, *, opacity=1):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self.opacity = opacity

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

    @property
    def clipping_planes(self):
        """A tuple of planes (abcd tuples) in world space. Points in
        space whose signed distance to the plane is negative are clipped
        (not rendered). Applies to the object to which this material is attached.
        """
        # todo: assumes singular plane
        return tuple(self.uniform_buffer.data["clipping_planes"])

    @clipping_planes.setter
    def clipping_planes(self, planes):
        if not isinstance(planes, (tuple, list)):
            raise TypeError("Clipping planes must be a list.")
        planes2 = []
        for plane in planes:
            if isinstance(plane, (tuple, list)) and len(plane) == 4:
                planes2.append(plane)
            # todo?: elif isinstance(plane, pga3.Plane):
            else:
                raise TypeError(
                    f"Each clipping plane must be an abcd tuple, not {plane}"
                )

        # todo: support multiple planes
        assert len(planes2) <= 1
        if len(planes2) == 0:
            self.uniform_buffer.data["clipping_planes"] = 0, 0, 0, 0
        else:
            self.uniform_buffer.data["clipping_planes"] = planes2[0]
        self.uniform_buffer.update_range(0, 1)

    # todo: clip_intersection
