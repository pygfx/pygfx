from ._base import WorldObject


class Line(WorldObject):
    """An object consisting of a line represented by a list of vertices
    (3D positions). Some materials will render the line as a continuous line,
    while other materials will consider each pair of points a segment.

    The picking info of a Line (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (float). Note that ``vertex_index`` is not integer;
    round the number to obtain the nearest vertex.
    """

    def __init__(self, geometry, material):
        super().__init__()
        self.geometry = geometry
        self.material = material
