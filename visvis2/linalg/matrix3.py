class Matrix3:
    def __init__(self) -> None:
        self.elements = [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ]

    def set(
        self,
        n11: float,
        n12: float,
        n13: float,
        n21: float,
        n22: float,
        n23: float,
        n31: float,
        n32: float,
        n33: float,
    ) -> "Matrix4":
        te = self.elements

        te[0] = n11
        te[3] = n12
        te[6] = n13
        te[1] = n21
        te[4] = n22
        te[7] = n23
        te[2] = n31
        te[5] = n32
        te[8] = n33

        return self
