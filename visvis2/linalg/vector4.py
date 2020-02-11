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

    def apply_matrix4(self, m: "Matrix4") -> "Vector4":
        x = self.x
        y = self.y
        z = self.z
        w = self.w
        e = m.elements

        self.x = e[0] * x + e[4] * y + e[8] * z + e[12] * w
        self.y = e[1] * x + e[5] * y + e[9] * z + e[13] * w
        self.z = e[2] * x + e[6] * y + e[10] * z + e[14] * w
        self.w = e[3] * x + e[7] * y + e[11] * z + e[15] * w

        return self
