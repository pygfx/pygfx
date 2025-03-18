import numpy as np


class Bounds:
    """Represent bounds using an axis-aligned bounding box (aabb) plus a radius.

    The radius represents a sphere centered at the aabb. Semantically,
    the bounds are the intersection of the box and the spere. This
    allows applying transforms without growing the bounds because of
    the box's corners.

    When the radius is None, or near-equal to half the diagonal of the aabb,
    the bounds respresents a box. The radius is at least the
    max(width,height,depth) of the aabb, i.e. only cutting off its corners.
    """

    __slots__ = ["aabb", "radius"]

    def __init__(self, aabb, radius=None):
        aabb = np.asarray(aabb)
        if aabb.shape != (2, 3):
            raise ValueError("aabb must be 2x3 array")
        self.aabb = aabb
        if radius is not None:
            min_radius = 0.5 * float(abs((aabb[1] - aabb[0]).max()))  # w/h/d
            max_radius = 0.5 * float(np.linalg.norm(aabb[1] - aabb[0]))  # diagonal
            if radius > 0.99 * max_radius:
                radius = None
            else:
                radius = max(radius, min_radius)
        self.radius = radius

    def __repr__(self):
        p1 = ", ".join(f"{i:0.4g}" for i in self.aabb[0])
        p2 = ", ".join(f"{i:0.4g}" for i in self.aabb[1])
        r = "without radius"
        if self.radius is not None:
            r = f"with radius {self.radius:0.4g}"
        return f"<Bounds aabb from ({p1}) to ({p2}) {r} at {hex(id(self))}>"

    def __eq__(self, other):
        return np.all(self.aabb == other.aabb) and self.radius == other.radius

    @property
    def center(self):
        aabb = self.aabb
        return tuple(0.5 * (aabb[0] + aabb[1]))

    @property
    def sphere(self):
        aabb = self.aabb
        radius = self.radius
        if radius is None:
            radius = 0.5 * float(np.linalg.norm(aabb[1] - aabb[0]))
        c = 0.5 * (aabb[0] + aabb[1])
        return float(c[0]), float(c[1]), float(c[2]), radius

    @classmethod
    def from_points(cls, points, calculate_radius=False):
        aabb, radius = points_to_bounds(points, calculate_radius)
        if aabb is None:
            return None
        return cls(aabb, radius)

    @classmethod
    def combine(cls, bounds_objects):
        pointsets = []
        for bounds in bounds_objects:
            if bounds is not None:
                pointsets.append(bounds_to_points(bounds.aabb, bounds.radius))
        points = np.vstack(pointsets)
        return cls.from_points(points[:, :3])

    def transform(self, matrix):
        points = bounds_to_points(self.aabb, self.radius)
        np.matmul(points, matrix, out=points)
        return self.from_points(points[:, :3])

    def intesects_point(self, xyz):
        raise NotImplementedError("TODO")

    def intesects_line(self, point, direction):
        raise NotImplementedError("TODO")


def points_to_bounds(points, calculate_radius=True):
    # Check points
    points = np.asarray(points)
    if not (points.ndim == 2 and points.shape[1] in (2, 3)):
        raise ValueError("Points must be a list of 2D or 3D points.")
    if points.shape[0] == 0:
        return None, None

    # Get aabb
    aabb = np.array([np.min(points, axis=0), np.max(points, axis=0)], dtype=float)
    if aabb.shape[1] == 2:
        aabb = np.column_stack([aabb, np.zeros((2, 1), float)])

    # If the aabb has nonfinite values, fix and recurse! The min/max above
    # is a very efficient way to detect whether there are nonfinite values.
    # So we don't have to first do a full check on the points themselves.
    if not np.isfinite(aabb).all():
        finite_mask = np.isfinite(points).all(axis=1)
        return points_to_bounds(points[finite_mask])

    radius = None
    if calculate_radius:
        # Find radius
        c = 0.5 * (aabb[0] + aabb[1])
        distances = np.linalg.norm(points - c, axis=1)
        radius = float(max(distances))
        # The radius only has effect if its smaller than half the diagonal
        diagonal = float(np.linalg.norm(aabb[1] - aabb[0]))
        if radius > 0.49 * diagonal:
            radius = None

    return aabb, radius


def bounds_to_points(aabb, radius):
    w, h, d = aabb[1] - aabb[0]

    if w == 0 and h == 0 and d == 0:
        # Single point
        points = aabb[0].copy().reshape(1, 3)
    elif d == 0:
        # Flat in z (like an image)
        if radius is None:
            # Init points
            points = np.full((4, 4), fill_value=1.0, dtype=float)
            points[:, 2] = aabb[0, 2]  # z
            # The corners of the aabb
            points[0::2, 0] = aabb[0, 0]  # x
            points[1::2, 0] = aabb[1, 0]
            points[0:2, 1] = aabb[0, 1]  # y
            points[2:4, 1] = aabb[1, 1]
        else:
            c = 0.5 * (aabb[0] + aabb[1])
            rx = min(0.707106 * radius, w / 2)
            ry = min(0.707106 * radius, h / 2)
            # Init points
            points = np.full((8, 4), fill_value=1.0, dtype=float)
            points[:, 0:3] = c
            # Corners, based on radius, clipped by aabb
            points[0:4:2, 0] += rx  # x
            points[1:4:2, 0] -= rx
            points[0:2, 1] += ry  # y
            points[2:4, 1] -= ry
            # The axis of the aabb
            points[4, 0] = aabb[0, 0]
            points[5, 0] = aabb[1, 0]
            points[6, 1] = aabb[0, 1]
            points[7, 1] = aabb[1, 1]
    else:
        # 3D
        if radius is None:
            # Init points
            points = np.full((8, 4), fill_value=1.0, dtype=float)
            # The corners of the aabb
            points[0::2, 0] = aabb[0, 0]  # x
            points[1::2, 0] = aabb[1, 0]
            points[0::4, 1] = aabb[0, 1]  # y
            points[1::4, 1] = aabb[0, 1]
            points[2::4, 1] = aabb[1, 1]
            points[3::4, 1] = aabb[1, 1]
            points[0:4, 2] = aabb[0, 2]  # z
            points[4:8, 2] = aabb[1, 2]
        else:
            c = 0.5 * (aabb[0] + aabb[1])
            rx = min(0.707106 * radius, w / 2)
            ry = min(0.707106 * radius, h / 2)
            rz = min(0.707106 * radius, d / 2)
            # Init points
            points = np.full((14, 4), fill_value=1.0, dtype=float)
            points[:, 0:3] = c
            # Corners, based on radius, clipped by aabb
            points[0:8:2, 0] += rx  # x
            points[1:8:2, 0] -= rx
            points[0:8:4, 1] += ry  # y
            points[1:8:4, 1] += ry
            points[2:8:4, 1] -= ry
            points[3:8:4, 1] -= ry
            points[0:4, 2] += rz  # z
            points[4:8, 2] -= rz
            # The aixs of the aabb
            points[8, 0] = aabb[0, 0]
            points[9, 0] = aabb[1, 0]
            points[10, 1] = aabb[0, 1]
            points[11, 1] = aabb[1, 1]
            points[12, 2] = aabb[0, 2]
            points[13, 2] = aabb[1, 2]

    return points
