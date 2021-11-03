from ._base import WorldObject


class Volume(WorldObject):
    """A volume represents a 3D image in space.

    The geometry for this object consists only of `geometry.grid`: a texture with the 3D data.

    The picking info of a Volume (the result of ``renderer.get_pick_info()``)
    will for most materials include ``voxel_index`` (tuple of 3 floats).
    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        size = tex.size
        x, y, z = [(v / 1048576) * s - 0.5 for v, s in zip(pick_value[1:], size)]
        return {"instance_index": 0, "voxel_index": (x, y, z)}
