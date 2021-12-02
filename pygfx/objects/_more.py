from ._base import WorldObject
from ..utils import unpack_bitfield


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
    ``vertex_index`` (int) and ``segment_coord`` (float, sub-segment coordinate).
    """


class Points(WorldObject):
    """An object consisting of points represented by vertices (3D positions).

    The picking info of a Points object (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (int) and ``point_coord`` (tuple of 2 float coordinates
    in logical pixels).
    """


class Mesh(WorldObject):
    """An object consisting of triangular faces, represented by vertices
    (3D positions) and an index that defines the connectivity.

    The picking info of a Mesh (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``instance_index`` (int), ``face_index`` (int), and ``face_coord``
    (tuple of 3 floats). The latter are the barycentric coordinates for
    each vertex of the face (with values 0..1).
    """


class Volume(WorldObject):
    """An object representing a 3D image in space (a volume).

    The geometry for this object consists only of `geometry.grid`: a texture with the 3D data.

    The picking info of a Volume (the result of ``renderer.get_pick_info()``)
    will for most materials include ``index`` (tuple of 3 int),
    and ``voxel_coord`` (tuple of float subpixel coordinates).
    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        # This should match with the shader
        _, *texcoords_encoded = unpack_bitfield(pick_value, 20, 14, 14, 14)
        size = tex.size
        x, y, z = [(v / 16384) * s - 0.5 for v, s in zip(texcoords_encoded, size)]
        ix, iy, iz = int(x + 0.5), int(y + 0.5), int(z + 0.5)
        return {
            "index": (ix, iy, iz),
            "voxel_coord": (x - ix, y - iy, z - iz),
        }
