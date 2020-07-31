import math


class Spherical:
    _2pi = 2 * math.pi
    _eps = 0.000001

    def __init__(
        self, radius: float = 1.0, phi: float = 0.0, theta: float = 0.0
    ) -> None:
        self.radius = radius
        self.phi = phi
        self.theta = theta

    def set_from_vector3(self, v: "Vector3") -> "Spherical":
        return self.set_from_cartesian_coords(v.x, v.y, v.z)

    def set_from_cartesian_coords(self, x: float, y: float, z: float) -> "Spherical":
        self.radius = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        if self.radius == 0:
            self.theta, self.phi = 0, 0
        else:
            self.theta = math.atan2(x, z)
            y /= self.radius
            if y < -1:
                y = -1
            elif y > 1:
                y = 1
            self.phi = math.acos(y)
        return self

    def make_safe(self) -> "Spherical":
        # restrict phi to (epsilon, pi-epsilon)
        self.phi = max(self._eps, min(math.pi - self._eps, self.phi))
        # restrict theta to (0, 2*pi)
        self.theta = self.theta % self._2pi
        return self
