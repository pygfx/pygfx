# from ..objects import WorldObject

# todo: all do plural or singular here!
from visvis2.objects import WorldObject
from visvis2.material import Material


class Renderer:
    """ Base class for other renderers. A renderer takes a figure,
    collect data that describe how it should be drawn, and then draws it.
    """

    pass


class RenderFunctionRegistry:
    """ A registry for render functions capable of rendering specific
    object-material combinations. This registry allows for a scalable
    plugin-like system for a renderer's capabilities.
    """

    def __init__(self):
        self._renderers = {}

    def register(self, wobject_cls, material_cls, func=None):
        """ Register a render function for the given combination of
        world object and material class. If func is not given or None,
        a function is returned that takes a single callable as argument,
        so this method can be used as a decorator.

        When a world object is being rendered, the renderer will select
        a render function based on the types of the world object and
        its material. The selection prioritizes more specialized
        classes. Therefore, custom render functions should be registered
        as specifically as possible to avoid clashes. Builtin functions
        can register "catch all" functions, e.g. providing a solution
        for all materials with a specific world object subclass, using
        with (SomeWorldObject, Material).
        """

        # Check types
        if not (isinstance(wobject_cls, type) and issubclass(wobject_cls, WorldObject)):
            raise TypeError(f"Expected WorldObject subclass, got {wobject_cls}.")
        if not (isinstance(material_cls, type) and issubclass(material_cls, Material)):
            raise TypeError(f"Expected Material subclass, got {material_cls}.")

        key = (wobject_cls, material_cls)

        # Define function to insert in our dict-tree (or "sparse martrix")
        def register_func(f):
            assert callable(f)
            if key in self._renderers:
                raise ValueError(
                    "Each combination of WorldObject and Material can only be registered once."
                )
            self._renderers[key] = f

        if func is not None:
            register_func(func)
        else:
            return register_func  # This call is probably a decorator

    def get_render_function(self, wobject):
        """ Get the render function for the given world object, based
        on the object's class, and the class of its material. Returns
        None if the object has no material, or a suitable renderer
        cannot be selected.
        """

        # Get and check world object type
        wobject_cls = type(wobject)
        if not isinstance(wobject, WorldObject):
            raise TypeError(f"Expected WorldObject instance, got {wobject_cls}.")

        # Get the material, if the object has it
        material = getattr(wobject, "material", None)
        if material is None:
            return None

        # Get and check the material type
        material_cls = type(material)
        if not isinstance(material, Material):
            raise TypeError(f"Expected Material instance, got {material_cls}.")

        # Try to select a renderer
        for cls1 in wobject_cls.mro():
            if issubclass(cls1, WorldObject):
                for cls2 in material_cls.mro():
                    if issubclass(cls2, Material):
                        key = cls1, cls2
                        f = self._renderers.get(key, None)
                        if f is not None:
                            return f

        return None
