from ._base import WorldObject


class Points(WorldObject):
    """An object consisting of points represented by vertices (3D positions).

    The picking info of a Points object (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (int).

    """

    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry
        self.material = material
