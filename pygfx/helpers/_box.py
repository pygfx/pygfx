import numpy as np

from .. import Geometry, Line, LineSegmentMaterial
from ..objects import WorldObject


class BoxHelper(Line):
    """A WorldObject that shows a box-shaped wireframe.

    Commonly used to visualize bounding boxes.

    Parameters
    ----------
    size : float
        The length of the box' edges in local space.
    thickness : float
        The thickness of the lines in (onscreen) pixels.
    color : Color
        The color of the box.

    """

    def __init__(self, size=1.0, thickness=1, color="white"):
        self._size = float(size)

        positions = np.array(
            [
                [0, 0, 0],  # bottom edges
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 0, 1],
                [0, 1, 0],  # top edges
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
                [0, 0, 0],  # side edges
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype="f4",
        )
        positions -= 0.5

        positions *= self._size

        geometry = Geometry(positions=positions)
        material = LineSegmentMaterial(thickness=thickness, color=color, aa=True)

        super().__init__(geometry, material)

    def set_transform_by_aabb(self, aabb, scale=1.0):
        """Align with axis aligned bounding box.

        The position and scale attributes will be configured such that the
        helper will match the given bounding box.

        Parameters
        ----------
        aabb : ndarray, [2, 3]
            The bounding box to align with. The two vectors represent the
            minimum and maximum coordinates of the axis-aligned bounding box.
        scale : float
            Scale multiplier of the final wireframe. Useful for adding margin to the
            box.

        """

        aabb = np.asarray(aabb)
        if aabb.shape != (2, 3):
            raise ValueError(
                "The given array does not appear to represent "
                "an axis-aligned bounding box, ensure "
                "the shape is (2, 3). Shape given: "
                f"{aabb.shape}"
            )

        diagonal = aabb[1] - aabb[0]
        center = aabb[0] + diagonal * 0.5
        full_scale = scale * diagonal / self._size

        self.local.position = center
        self.local.scale = full_scale

    def set_transform_by_object(self, object: WorldObject, space="world", scale=1.0):
        """Align with WorldObject.

        Set the position and scale attributes based on the bounding box of
        another object.

        Parameters
        ----------
        object : WorldObject
            The object to wrap inside this wireframe.
        space : str
            If "world", the wire will be aligned to the world's axes. If
            "local", the wire will be aligned to the local axes.
        scale : float
            Scale multiplier of the final wireframe. Useful for adding margin to the
            box.

        Examples
        --------

        World-space bounding box visualization::

            box = gfx.BoxHelper()
            box.set_transform_by_object(mesh)
            scene.add(box)

        Local-space bounding box visualization::

            box = gfx.BoxHelper()
            box.set_transform_by_object(mesh, space="local")
            mesh.add(box)

        """

        aabb = None
        if space not in {"world", "local"}:
            raise ValueError(
                'Space argument must be either "world"'
                f'or "local". Given value: {space}'
            )
        if space == "world":
            aabb = object.get_world_bounding_box()
        elif space == "local" and object.geometry is not None:
            aabb = object.geometry.get_bounding_box()
        if aabb is None:
            raise ValueError(
                "No bounding box could be determined "
                "for the given object, it (and its "
                "children) may not define any geometry"
            )
        self.set_transform_by_aabb(aabb, scale)

    def get_world_bounding_box(self):
        return None

    def get_world_bounding_sphere(self):
        return None
