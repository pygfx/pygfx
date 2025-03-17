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
