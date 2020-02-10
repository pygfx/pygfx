from ._world_object import WorldObject


class Mesh(WorldObject):
    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry
        self.material = material

    def get_renderer_info_wgpu(self):
        d = {}
        # d.update(self.geometry.get_buffer_info())
        d.update(self.material.get_renderer_info_wgpu())
        return d

    def get_renderer_info_svg(self):
        d = {}
        # d.update(self.geometry.get_buffer_info())
        d.update(self.material.get_renderer_info_svg())
        return d
