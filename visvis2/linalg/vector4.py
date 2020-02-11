from .utils import clamp
from .quaternion import Quaternion


class Vector4:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, w: float = 0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __repr__(self) -> str:
        return f"Vector4({self.x}, {self.y}, {self.z}, {self.w})"

    def set(self, x: float, y: float, z: float, w: float) -> "Vector4":
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        return self
