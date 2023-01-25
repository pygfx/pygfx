from ..objects import WorldObject
from ..materials import Material


class Renderer:
    """Renderer base class."""

    pass


class RenderFunctionRegistry:
    """Storage for available rendering functions.

    A registry for render functions capable of rendering specific
    object-material combinations. This registry allows for a scalable
    plugin-like system for a renderer's capabilities.

    """

    def __init__(self):
        self._store = {}
        self._known_classes = set([WorldObject, Material])

    def register(self, wobject_cls, material_cls, func):
        """Register a render function for the given combination of
        world object and material class.

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
        if not callable(func):
            raise TypeError("Third argument must be a callable, not {func}")

        # Store func with a key derived from the classes
        key = (wobject_cls, material_cls)
        if key in self._store:
            raise ValueError(
                "Each combination of WorldObject and Material can only be registered once."
            )
        self._store[key] = func
        self._known_classes.add(wobject_cls)
        self._known_classes.add(material_cls)

    def get_render_function(self, wobject):
        """Get the render function for the given world object, based
        on the object's class, and the class of its material. The behavior
        is similar to ``isinstance``; providing an instance of a custom
        subclasses should not affect the selection. Returns None if the
        object has no material, or a suitable renderer cannot be
        selected.
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

        # Get WorldObject mro, but stripped to classes of interest
        wobject_classes = wobject_cls.mro()
        while wobject_classes[-1] is not WorldObject:
            wobject_classes.pop(-1)
        while wobject_classes[0] not in self._known_classes:
            wobject_classes.pop(0)

        # Get Material mro, but stripped to classes of interest
        material_classes = material_cls.mro()
        while material_classes[-1] is not Material:
            material_classes.pop(-1)
        while material_classes[0] not in self._known_classes:
            material_classes.pop(0)

        # Try to select a renderer. There is a double-loop here, but
        # both iters should be short, especially with the stripping
        # above. An alternative implementation would be to walk over
        # all registry entries and use isinstance, but the number of
        # render functions will (over time) exceed the depth of
        # subclasses that is traversed here.
        for cls1 in wobject_classes:
            for cls2 in material_classes:
                key = cls1, cls2
                f = self._store.get(key, None)
                if f is not None:
                    return f

        return None
