from ._base import Material


class MeshBasicMaterial(Material):
    def __init__(self):
        self.color = (255.0, 0.0, 0.0)
