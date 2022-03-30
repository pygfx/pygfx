from ..objects import WorldObject, Scene
from ..cameras import Camera


class View:
    """A convenience class that contains:
    * A scene.
    * A camera to render the scene with.
    * A viewport (rectangular region) to render the scene to.
    * A reference to the renderer.
    """

    def __init__(self, renderer, camera=None, scene=None, viewport=None):

        self._renderer = renderer
        self.scene = scene or Scene()
        self._camera = None
        if camera:
            self._camera = camera
        self.viewport = viewport

    @property
    def renderer(self):
        """The renderer to render the scene."""
        return self._renderer

    @property
    def scene(self):
        """The scene for this view."""
        return self._scene

    @scene.setter
    def scene(self, scene):
        if not isinstance(scene, WorldObject):
            raise TypeError("View.scene must be a world object.")
        self._scene = scene

    @property
    def camera(self):
        if self._camera is None:
            for child in self._scene.children:
                if isinstance(child, Camera):
                    self._camera = child
                    break
        if self._camera is None:
            raise RuntimeError("View has no camera set")
        return self._camera

    @camera.setter
    def camera(self, camera):
        if not isinstance(camera, Camera):
            raise TypeError("View.camera must be a camera.")
        self._camera = camera

    @property
    def viewport(self):
        """The rectangular region to render to, as (x, y, w, h).
        Can be set to `None` to follow the renderer's full size.
        """
        if self._viewport is None:
            return (0, 0) + self.renderer.logical_size
        else:
            return self._viewport

    @viewport.setter
    def viewport(self, value):
        if value is None:
            self._viewport = None
        else:
            if not (
                len(value) == 4 and all(isinstance(i, (int, float)) for i in value)
            ):
                raise ValueError("view.wiewport must consist of 4 numbers.")
            self._viewport = tuple(float(i) for i in value)

    @property
    def logical_size(self):
        """The size of the view in logical pixels."""
        vp = self.viewport
        return vp[2], vp[3]

    def in_viewport(self, x, y):
        """Get wether the given location (in logical pixels, relative
        to the canvas) is withing the viewport.
        """
        vp = self.viewport
        return vp[0] <= x <= vp[0] + vp[2] and vp[1] <= y <= vp[1] + vp[3]
