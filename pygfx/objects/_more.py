from ._base import WorldObject


class Group(WorldObject):
    """A Group is almost identical to a WorldObject. Its purpose is
    to make working with groups of objects syntactically clearer.
    """


class Scene(Group):
    """The scene is a WorldObject that represents the root of a scene graph.
    It hold certain attributes that relate to the scene, such as the background color,
    fog, and environment map. Camera's and lights can also be part of a scene.
    """


class Background(WorldObject):
    """An object representing a scene background.
    Can be e.g. a gradient, a static image or a skybox.
    """

    def __init__(self, geometry=None, material=None):
        if geometry is not None and material is None:
            raise TypeError("You need to instantiate using Background(None, material)")
        super().__init__(None, material)


class Line(WorldObject):
    """An object consisting of a line represented by a list of vertices
    (3D positions). Some materials will render the line as a continuous line,
    while other materials will consider each pair of points a segment.

    The picking info of a Line (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (float). Note that ``vertex_index`` is not integer;
    round the number to obtain the nearest vertex.
    """


class Points(WorldObject):
    """An object consisting of points represented by vertices (3D positions).

    The picking info of a Points object (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (int).

    """


class Mesh(WorldObject):
    """An object consisting of triangular faces, represented by vertices
    (3D positions) and an index that defines the connectivity.

    The picking info of a Mesh (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``instance_index`` (int), ``face_index`` (int), and ``face_coords``
    (tuple of 3 floats). The latter are the barycentric coordinates for
    each vertex of the face (with values 0..1).
    """


class Volume(WorldObject):
    """An object representing a 3D image in space (a volume).

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
