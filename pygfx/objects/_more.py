import pylinalg as la

from ._base import WorldObject
from ..utils import unpack_bitfield
from ..utils.transform import AffineBase, callback


class Group(WorldObject):
    """A group of objects.

    A Group is useful when manipulating the scene graph as children can be
    jointly moved/scaled/rotated. It has no visual properties.

    Parameters
    ----------
    visible : bool
        If true, the object and its children are visible.
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def __init__(self, *, visible=True):
        super().__init__(visible=visible)


class Scene(Group):
    """Root of the scene graph.

    The scene holds scene-level information (background color, fog, environment
    map) as well as all objects that take part in the rendering process as
    either direct or indirect children/nested objects.

    """

    def __init__(self):
        super().__init__()


class Background(WorldObject):
    """The scene's background.

    Can be e.g. a gradient, a static image or a skybox.

    Parameters
    ----------
    geometry : Geometry
        Must be ``None``. Exists for compliance with the generic WorldObject
        API.
    material : Material
        The material to use when rendering the background.
    kwargs : Any
        Additional kwargs are forwarded to the object's :class:`base class
        <pygfx.objects.WorldObject>`.

    """

    def __init__(self, geometry=None, material=None, **kwargs):
        if geometry is not None and material is None:
            raise TypeError("You need to instantiate using Background(None, material)")
        super().__init__(None, material, **kwargs)


class Line(WorldObject):
    """An object representing a line using a list of vertices (3D positions).

    Some materials will render the line as a continuous line, while other materials
    will consider each pair of consecutive points a segment.

    The picking info of a Line (the result of ``renderer.get_pick_info()``) will
    for most materials include ``vertex_index`` (int) and ``segment_coord``
    (float, sub-segment coordinate).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """


class Points(WorldObject):
    """A point cloud.

    An object consisting of points represented by vertices (3D positions).

    The picking info of a Points object (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``vertex_index`` (int) and ``point_coord`` (tuple of 2 float coordinates in
    logical pixels).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """


class Mesh(WorldObject):
    """A mesh.

    An object consisting of triangular faces represented by a set of vertices
    (3D positions) and a set of vertex indices indicating which vertex triplets
    form mesh triangles.

    The picking info of a Mesh (the result of ``renderer.get_pick_info()``) will
    for most materials include ``instance_index`` (int), ``face_index`` (int),
    and ``face_coord`` (tuple of 3 floats). The latter are the barycentric
    coordinates for each vertex of the face (with values 0..1).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def _wgpu_get_pick_info(self, pick_value):
        values = unpack_bitfield(
            pick_value, wobject_id=20, index=26, coord1=6, coord2=6, coord3=6
        )
        face_index = values["index"]
        face_coord = [
            values["coord1"] / 63,
            values["coord2"] / 63,
            values["coord3"] / 63,
        ]
        if (
            self.geometry.indices.data is not None
            and self.geometry.indices.data.shape[-1] == 4
        ):
            triangle_index = face_index % 2
            face_index = face_index // 2
            if triangle_index == 0:
                # The sub indices are 0, 1, 2, so we just add the zero for index 3.
                face_coord.append(0.0)
            else:
                # The sub indices are 0, 2, 3. The index 3 uses the
                # face_coord slot of index 1, (see meshshader.py), so
                # we put that at the end and put a zero in its place.
                face_coord = face_coord[0], 0.0, face_coord[2], face_coord[1]

        return {"face_index": face_index, "face_coord": tuple(face_coord)}


class Image(WorldObject):
    """A 2D image.

    The geometry for this object consists only of ``geometry.grid``: a
    texture with the 2D data.

    If no colormap is applied to the material, the data are interpreted as
    colors in sRGB space. To use physical space instead, set the texture's
    colorspace property to ``"physical"``.

    The picking info of an Image (the result of ``renderer.get_pick_info()``)
    will for most materials include ``index`` (tuple of 2 int), and
    ``pixel_coord`` (tuple of float subpixel coordinates).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, x=22, y=22)
        x = values["x"] / 4194303 * tex.size[0] - 0.5
        y = values["y"] / 4194303 * tex.size[1] - 0.5
        ix, iy = int(x + 0.5), int(y + 0.5)
        return {
            "index": (ix, iy),
            "pixel_coord": (x - ix, y - iy),
        }


class Volume(WorldObject):
    """A 3D image.

    The geometry for this object consists only of ``geometry.grid``: a texture
    with the 3D data.

    The picking info of a Volume (the result of ``renderer.get_pick_info()``)
    will for most materials include ``index`` (tuple of 3 int), and
    ``voxel_coord`` (tuple of float subpixel coordinates).

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    def _wgpu_get_pick_info(self, pick_value):
        tex = self.geometry.grid
        if hasattr(tex, "texture"):
            tex = tex.texture  # tex was a view
        # This should match with the shader
        values = unpack_bitfield(pick_value, wobject_id=20, x=14, y=14, z=14)
        texcoords_encoded = values["x"], values["y"], values["z"]
        size = tex.size
        x, y, z = [(v / 16383) * s - 0.5 for v, s in zip(texcoords_encoded, size)]
        ix, iy, iz = int(x + 0.5), int(y + 0.5), int(z + 0.5)
        return {
            "index": (ix, iy, iz),
            "voxel_coord": (x - ix, y - iy, z - iz),
        }


class Text(WorldObject):
    """A text.

    See :class:``pygfx.TextGeometry`` for details.

    Parameters
    ----------
    geometry : TextGeometry
        The data defining the glyphs that make up the text.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : int
        The render order (when applicable for the renderer's blend mode).
    render_mask : str
        Determines the render passes that the object is rendered in. It's
        recommended to let the renderer decide, using "auto".
    position : Vector
        The position of the object in the world. Default (0, 0, 0).

    """

    uniform_type = dict(
        WorldObject.uniform_type,
        rot_scale_transform="4x4xf4",
    )

    def __init__(
        self,
        geometry=None,
        material=None,
        *,
        visible=True,
        render_order=0,
        render_mask="auto"
    ):
        super().__init__(
            geometry,
            material,
            visible=visible,
            render_order=render_order,
            render_mask=render_mask,
        )

        # calling super from callback is possible, but slow so we register it as a second callback instead
        self.world.on_update(super()._update_uniform_buffers)

    @callback
    def _update_uniform_buffers(self, transform: AffineBase):
        # super()._update_uniform_buffers(transform)
        # When rendering in screen space, the world transform is used
        # to establish the point in the scene where the text is placed.
        # The only part of the local transform that is used is the
        # position. Therefore, we also keep a transform containing the
        # local rotation and scale, so that these can be applied to the
        # text in screen coordinates.
        matrix = la.mat_compose((0, 0, 0), self.local.rotation, self.local.scale)
        self.uniform_buffer.data["rot_scale_transform"] = matrix.T
