MACHINE_EPSILON = (
    7.0 / 3 - 4.0 / 3 - 1
)  # the difference between 1 and the smallest floating point number greater than 1


def clamp(x: float, left: float, right: float) -> float:
    return max(left, min(right, x))
