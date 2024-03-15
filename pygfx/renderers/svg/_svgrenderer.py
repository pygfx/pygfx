import os
import io
import pylinalg as la

from ...utils.renderfunctionregistry import RenderFunctionRegistry

from ...objects import WorldObject
from ...cameras import Camera
from .. import Renderer


registry = RenderFunctionRegistry()


def register_svg_render_function(wobject_cls, material_cls):
    """Decorator for SVG rendering functions.

    Parameters
    ----------
    wobject_cls : WorldObject
        The world object that this function knows how to render.
    material_cls : Material
        The world object that this function knows how to render.

    """

    def _register_svg_render_function(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_svg_render_function


class SvgRenderer(Renderer):
    """Turns scenes into SVG images.

    Notice: this renderer is just an experimental stub for now.

    Parameters
    ----------
    width : int
        The width of the resulting image.
    height : int
        The height of the resulting image.
    filename : str
        The name of the location to which to write the image.

    """

    def __init__(self, width, height, filename):
        self._width = width
        self._height = height

        # todo: also support writing to file-like object
        if filename.startswith("~"):
            filename = os.path.expanduser(filename)
        self._filename = filename

    def render(self, scene: WorldObject, camera: Camera):
        """Render the scene to a file."""

        # Ensure that matrices are up-to-date
        camera.update_projection_matrix()

        # Get the sorted list of objects to render (guaranteed to be visible and having a material)
        q = self.get_render_list(scene, camera.camera_matrix)

        # Init the svg file
        f = io.StringIO()
        f.write(
            f"<svg width='{self._width}' height='{self._height}' xmlns='http://www.w3.org/2000/svg'>\n"
        )

        # Render each object (that we can render)
        for wobject in q:
            renderfunc = registry.get_render_function(wobject)
            if renderfunc is not None:
                res = renderfunc(wobject)
                if isinstance(res, str):
                    f.write(res)
                elif isinstance(res, list):
                    for line in res:
                        f.write(line)

        # Finish the svg file and write actual file handle.
        f.write("\n</svg>\n")
        with open(self._filename, "wb") as f2:
            f2.write(f.getvalue().encode())

    def get_render_list(self, scene: WorldObject, proj_screen_matrix):
        """Given a scene object, get a list of objects to render."""

        # start by gathering everything that is visible and has a material
        q = []

        def visit(wobject):
            nonlocal q
            if wobject.visible and hasattr(wobject, "material"):
                q.append(wobject)

        scene.traverse(visit)

        # next, sort them from back-to-front
        def sort_func(wobject: WorldObject):
            z = la.vec_transform(wobject.world.position, proj_screen_matrix)[2]
            return wobject.render_order, z

        return list(sorted(q, key=sort_func))
