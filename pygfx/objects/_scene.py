from ._base import WorldObject


class Scene(WorldObject):
    """The scene is a WorldObject that represents the root of a scene graph.
    It hold certain attributes that relate to the scene, such as the background color,
    fog, and environment map. Camera's and lights can also be part of a scene.
    """


class Background(WorldObject):
    """An object representing a scene background.
    Can be e.g. a gradient, a static image or a skybox.
    """

    def __init__(self, material):
        super().__init__()
        self.material = material
