import os
import io

from .. import Renderer, RenderFunctionRegistry

from ...objects import WorldObject
from ...cameras import Camera
from ...linalg import Matrix4, Vector3


registry = RenderFunctionRegistry()


def register_svg_render_function(wobject_cls, material_cls):
    """Decorator to register an SVG render function."""

    def _register_svg_render_function(f):
        registry.register(wobject_cls, material_cls, f)
        return f

    return _register_svg_render_function


class SvgRenderer(Renderer):
    """A renderer that generates an SVG."""

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
        scene.update_matrix_world()
        camera.update_matrix_world()  # camera may not be a member of the scene
        camera.update_projection_matrix()

        # Get the sorted list of objects to render (guaranteed to be visible and having a material)
        proj_screen_matrix = Matrix4().multiply_matrices(
            camera.projection_matrix, camera.matrix_world_inverse
        )
        q = self.get_render_list(scene, proj_screen_matrix)

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

    def get_render_list(self, scene: WorldObject, proj_screen_matrix: Matrix4):
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
            z = (
                Vector3()
                .set_from_matrix_position(wobject.matrix_world)
                .apply_matrix4(proj_screen_matrix)
                .z
            )
            return wobject.render_order, z

        return list(sorted(q, key=sort_func))
