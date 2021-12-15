from ..objects._base import ResourceContainer
from ..utils import array_from_shadertype
from ..resources import Buffer


class Material(ResourceContainer):
    """The base class for all materials.
    Materials define how an object is rendered, subject to certain properties.
    """

    uniform_type = dict(
        opacity="f4",
        clipping_planes="0*4xf4",  # array<vec4<f32>,3>
    )

    def __init__(self, *, opacity=1, clipping_planes=None, clipping_mode="any"):
        super().__init__()

        # Compose complete uniform type
        self.uniform_type = {}
        for cls in reversed(self.__class__.mro()):
            self.uniform_type.update(getattr(cls, "uniform_type", {}))

        # Create matching uniform buffer
        self.uniform_buffer = Buffer(array_from_shadertype(self.uniform_type))

        self.opacity = opacity
        self.clipping_planes = clipping_planes or []
        self.clipping_mode = clipping_mode

    def _set_size_of_uniform_array(self, key, new_length):
        """Resize the given array field in the uniform struct if the
        current length does not match the given length. This adjusts
        the uniform type, creates a new buffer from that, and copies
        all data except that of the given field.

        Resetting the uniform buffer will bump the rev and thus trigger
        a pipeline rebuild for all objects that this material is
        attached to.
        """
        current_length = self.uniform_buffer.data[key].shape[0]
        if new_length == current_length:
            return  # early exit

        format = self.uniform_type[key]
        assert "*" in format, f"uniform field {key} '{format}' is not an array"
        n, _, subtype = format.partition("*")
        assert int(n) >= 0

        # Adjust type definition (note that this is originally a class attr)
        self.uniform_type[key] = f"{new_length}*{subtype}"
        # Recreate buffer
        data = self.uniform_buffer.data
        self.uniform_buffer = Buffer(array_from_shadertype(self.uniform_type))
        # Copy data
        for k in data.dtype.names:
            if k != key:
                self.uniform_buffer.data[k] = data[k]

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
        non-opaque fragments, their alphas are simply scaled by this value.
        """
        return float(self.uniform_buffer.data["opacity"])

    @opacity.setter
    def opacity(self, value):
        value = min(max(float(value), 0), 1)
        if (value == 1) != (self.uniform_buffer.data["opacity"] == 1):
            self._bump_rev()  # rebuild pipeline if this becomes opaque/transparent
        self.uniform_buffer.data["opacity"] = value
        self.uniform_buffer.update_range(0, 1)

    @property
    def clipping_planes(self):
        """A tuple of planes (abcd tuples) in world space. Points in
        space whose signed distance to the plane is negative are clipped
        (not rendered). Applies to the object to which this material is attached.
        """
        return [
            tuple(float(f) for f in plane.flat)
            for plane in self.uniform_buffer.data["clipping_planes"]
        ]

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
                # Error
                if isinstance(plane, (int, float)) and len(planes) == 4:
                    raise TypeError(
                        f"Looks like you passed one clipping plane instead of a list."
                    )
                else:
                    raise TypeError(
                        f"Each clipping plane must be an abcd tuple, not {plane}"
                    )

        # Apply
        # Note that we can create a null-plane using (0, 0, 0, 0)
        self._set_size_of_uniform_array("clipping_planes", len(planes2))
        for i in range(len(planes2)):
            self.uniform_buffer.data["clipping_planes"][i] = planes2[i]
        self.uniform_buffer.update_range(0, 1)

    @property
    def clipping_mode(self):
        """Set the behavior for multiple clipping planes: "ANY" or "ALL".
        If this is "ANY" (the default) a fragment is discarded
        if it is clipped by any clipping plane. If this is "ALL", a
        fragment is discarded only if it is clipped by *all* of the
        clipping planes.
        """
        return self._clipping_mode

    @clipping_mode.setter
    def clipping_mode(self, value):
        mode = value.upper()
        if mode in ("ANY", "ALL"):
            self._clipping_mode = mode
        else:
            raise ValueError(f"Unexpected clipping_mode: {value}")
        self._bump_rev()
