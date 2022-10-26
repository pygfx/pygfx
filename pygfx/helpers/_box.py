import numpy as np

from .. import Geometry, Line, LineSegmentMaterial


class BoxHelper(Line):
    """A line box object. Commonly used to visualize bounding boxes.

    Parameters:
        size (float): The length of the box' edges (default 1).
        thickness (float): the thickness of the lines (default 1 px).
        color (Color): the color of the box.
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
        """Set the position and scale attributes
        based on a given bounding box.

        Parameters:
            aabb (ndarray): The position and scale attributes
                will be configured such that the helper
                will match the given bounding box. The array
                is expected to have shape (2, 3), where the
                two vectors represent the minimum and maximum
                coordinates of the axis-aligned bounding box.
            scale (float): the relative size of the box (oversize for
                a bit of margin).
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

        self.position.set(*center)
        self.scale.set(*full_scale)

    def set_transform_by_object(self, object, space="world", scale=1.0):
        """Set the position and scale attributes
        based on the bounding box of another object.

        Parameters:
            object (WorldObject): The position and scale attributes
                will be configured such that the helper
                will match the bounding box of the given object.
            space (string, optional): If set to "world"
                (the default) the world space bounding box will
                be used as reference. If equal to "local", the
                object's local space bounding box of its geometry
                will be used instead.
            scale (float): the relative size of the box (oversize for
                a bit of margin).

        :Examples:

        World-space bounding box visualization:

        .. code-block:: py

            box = gfx.BoxHelper()
            box.set_transform_by_object(mesh)
            scene.add(box)

        Local-space bounding box visualization:

        .. code-block:: py

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
            aabb = object.geometry.bounding_box()
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
