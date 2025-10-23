---------------------
Transparency in PyGfx
---------------------

When rendering objects with PyGfx, the shader calculates the color and alpha
value for each fragment (i.e. output pixel) and then writes these to the output
texture (the render target).

In the simplest case, the object is solid (i.e. opaque), and the colors simply overwrite the
colors in the output texture. In other cases, the colors are combined with
those of earlier-rendered objects, and there are multiple ways to do this.


Alpha
-----

Transparency is expressed using an ``alpha`` value. This can be the fourth
component in an RGBA color, e.g. in a color property, colormap or image texture. But it
can also be an explicit value, e.g. ``material.opacity``.

Although alpha values are commonly used to represent transparency, this is not always
the case; they can e.g. be used as the reference value in alpha testing.

In any case, the alpha value represents a weight for how the object is combined with
other objects, and its application is fully defined by ``material.alpha_config``.
This config allows a high degree of control, but the majority of cases can be
captured with a handful of presets that we call *alpha modes*.


Controlling transparency
------------------------

Transparency is a notoriously tricky topic in 3D rendering. Methods that produce
good results out of the box do exist, but are slow and/or consume considerably more memory.
In PyGfx we provide a few different methods that are relatively lean.

There are 3 levels of control:

1. Use the default ``material.alpha_mode = "auto"``:

In this mode, solid objects blend and write depth. Objects that
are opaque (alpha=1) are rendered correctly, as are objects that are
(partially) transparent, as long as the objects
don't intersect (in other words, they can be sorted based on their distance from the
camera).

2. Set ``material.alpha_mode`` to a preset string:

These provide configurations for common cases. Examples are "solid",
"blend", "weighted_blend", "dither", and several more. See below for a full list of options and descriptions.

3. Set the ``alpha_config`` dictionary to have full control:

In this advanced approach you can choose between four methods ("opaque",
"blended", "weighted", "stochastic"), which each have a set of options.
You probably also want to set ``material.depth_write``, and maybe
``material.render_queue`` and/or ``ob.render_order``.


Alpha what?
-----------

The material has multiple properties related to transparency. Let's list them for clarity:

These three belong together (setting one also sets the others):

* ``material.alpha_config``: a dict that defines the alpha behaviour in detail.
* ``material.alpha_mode``: a convenient way to set ``alpha_config`` using a preset string.
* ``material.alpha_method`` (readonly): a shorthand for ``alpha_config['method']``. Can be either "opaque", "blended", "weighted", or "stochastic".

Further alpha props:

* ``material.opacity``: an alpha multiplier applied to all fragments of the object.
* ``material.alpha_test``: the value to compare a fragment's alpha value with to determine if it should be discarded. For things like cut-outs.
* ``material.alpha_compare``: The comparison function for the alpha test, e.g. ``<``, ``<=``, or ``>``.

Some of these properties affect other properties:

* If the ``material.render_queue`` is not set, it is derived from ``alpha_mode``, ``alpha_method`` and ``alpha_test``.
* If the ``material.depth_write`` is not set, it is derived from ``alpha_mode`` and ``alpha_method`` (``depth_write = alpha_mode=='auto' or alpha_method in ('opaque', 'stochastic')``).


A quick guide to select alpha mode
----------------------------------

The trick to prevent/solve most problems related to transparency is to either make sure that
the order in which objects are rendered is correct, or to select a mode in which the order does not matter.

Pygfx sorts objects by their distance to the camera. See ``renderer.sort_objects``. But this may not suffice.
To control the order in which objects are rendered, you can set ``material.render_queue``, ``ob.render_order``, and ``material.depth_write``.
These are all explained in this document.

Your scene is 2D
================

If your scene is 2D, you can probably use ``alpha_mode`` "blend". It is
advisable to use the z-dimension to "layer" your objects, which will help the
renderer to sort the objects. The "auto" mode will also work (but writes to the depth buffer).

Assuming that objects don't intersect, you can turn on ``material.aa`` for text, lines, and points,
for prettier results.

If you need performance, you can use mode "solid" for objects that are known to
be opaque. (Though note that mode "solid" disables the use ``material.aa``).

Your scene is 3D
================

If your scene is 3D, and you only have a few transparent objects, it's probably best/easiest
to render solid objects with mode "solid", and the few transparent objects with mode "blend".

If you have a transparent object that intersects other transparent objects or
itself, the above will not produce satisfactory results, because there is no
"correct" order.

Maybe you can split the objects into multiple smaller objects, so that they no
longer intersect and can be unambiguously sorted by the renderer, based on their
distance to the camera.

Another option to deal with complex transparent geometry is to use mode
"dither". This produces excellent results from a technical perspective, but produces somewhat noisy images.
Alternatively, the mode "bayer" is also stochastic, but produces patterns that don't look so noisy.
You can also apply one of these modes to *some* of the transparent objects, e.g. the ones with the more complex geometry, and use
"blend" for the rest.

Similarly, the mode "weighted_blend" produces the same result indepent of object order. The
pixel values of all transparent objects are combined in a way that's independent
of their order or depth. This produces "clear" images, although not really correct, and
intersections of transparent objects are not visible.

You can also consider making some objects "solid" instead of transparent to
avoid transparency problems.

If you deal with objects that have both opaque and semi-transparent regions:
the "auto" mode may render these correctly, assuming objects can be sorted without intersections.
Otherwise mode "dither" can handle these cases well.


Alpha modes
-----------

In Pygfx, ``material.alpha_mode`` can be used to define how the alpha value of an object's fragment
is used to combine it with the output texture. There are a range of values to choose from, divided over four different methods:

Method "opaque" (overwrites the value in the output texture):

* "solid": alpha is ignored.
* "solid_premul": the alpha is multiplied with the color (making it darker).

Method "blended" (per-fragment blending, a.k.a. compositing):

* "auto": classic alpha blending, with ``depth_write`` defaulting to True.
* "blend": classic alpha blending using the over-operator.
* "add": additive blending that adds the fragment color, multiplied by alpha.
* "subtract": subtractive blending that removes the fragment color.
* "multiply": multiplicative blending that multiplies the fragment color.

Method "weighted" (order independent blending):

* "weighted_blend": weighted blended order independent transparency.
* "weighted_solid": fragments are combined based on alpha, but the final alpha is always 1. Great for e.g. image stitching.

Method "stochastic" (alpha represents the chance of a fragment being visible):

* "dither": stochastic transparency with blue noise.
* "bayer": stochastic transparency with a Bayer pattern.


Alpha methods
-------------

Most users don't have to worry much about what the alpha-methods mean. Though it's good to understand
that the "opaque" and "stochastic" methods produce opaque fragments, and by default have ``depth_write=True``.
The renderer sorts these objects front-to-back to avoid overdraw (for performance).

In contrast, the "blended" and "weighted" methods result in semi-transparent fragments,
and by default have ``depth_write=False``. The renderer sorts these object back-to-front to
improve the chance of correct blending.

**Alpha method 'opaque'** represents no transparency. The fragment color
overwrites the value in the output texture. A very common method in render engines.

**Alpha method 'blended'** represents alpha compositing: a common method in
render engines in which objects are combined on a per-fragment basis. The
object's fragment color and the current color in the output texture are blended
using a configurable operator. There are several common blending configurations,
the most-used being the "over operator" (also known as normal blending). When
blending is used, the result will depend on the order in which the objects are
rendered.

**Alpha method 'weighted'** represents (variants of) weighted blended order
independent transparency. The order of objects does not matter for the
end-result. One use-case being order independent transparency (OIT).
The order-independent property is advantageous in some use-cases, but produces
unfavourable results in others. It's use extends beyond transparency though, and
can also be used for e.g. image stiching.

**Alpha method 'stochastic'** represents stochastic transparency. The alpha
represents the chance of a fragment being visible (i.e. not discarded). Visible
fragments are opaque. This blend method is less common, but has interesting properties.
Although the result has a somewhat noisy appearance, it handles transparency perfectly,
capable of rendering multiple layers of transparent objects, and correctly handling
objects that have a mix of opaque and transparent fragments.


Alpha config
------------

The ``material.alpha_config`` is a dictionary that fully describes how the combining based on alpha occurs.
This dictionary has at least two keys: the 'method' and 'mode'. It has additional keys for the options
available for the used method. The different presets represent common combinations of these options.

Most users just set ``material.alpha_mode`` which implicitly sets
``material.alpha_method`` and ``material.alpha_config``. In advanced/special cases, users can set the
``material.alpha_config`` directly to take full control over all available
options. In this case the 'mode' field and ``material.alpha_mode`` become "custom".


Render queue
------------

The ``material.render_queue`` is an integer that represents the group that the renderer uses to sort objects.
The property is intended for advanced use; it is determined automatically
based on ``alpha_method``, ``depth_write`` and ``alpha_test``. Its value can be any integer between 1 and 4999,
and it comes with the following 'builtin' values:

* 1000: background.
* 2000: opaque non-blending objects.
* 2400: opaque objects with a discard based on alpha (i.e. using ``alpha_test`` or "stochastic" alpha-mode).
* 2600: objects with alpha-mode 'auto'.
* 3000: transparent objects.
* 4000: overlay.

These values are not accessible as enums because that would inhibit assignment of custom values. The set value
also affects behaviour: objects with ``render_queue`` between 1501 and 2500 are sorted front-to-back. Otherwise objects are sorted back-to-front.


Render order
------------

The ``object.render_order`` is a float that allows users to more precisely
control the order in which objects are rendered with respect to other objects in
the same ``render_queue``. You typically don't need this, but when you do, it's
good that you can. The value applies to the object and its children.


How the renderer sorts objects
------------------------------

The order in which objects are rendered is:

1. the ``material.render_queue``.
2. the effective ``object.render_order``.
3. the distance to camera (if ``renderer.sort_objects==True``).
4. the position of the object in the scene graph.

In step 3, objects are either sorted front-to-back if render_queue is between 1501 and 2500, and back-to-front otherwise. Objects with alpha-method 'weighted' are not sorted by depth.

Even with this sorting, objects can still intersect other objects (and themselves).
To prevent drawing the (parts of) objects that are occluded by other objects, a depth buffer is used.


Depth buffer
------------

The depth buffer is a texture of the same size as the color output texture, that
stores the distance from the camera of the last drawn fragment. If an object
has ``material.depth_test = True``, fragments that would be further from the
camera (i.e. are occluded by another object) will not be drawn. The ``material.depth_test`` is True by default.

One can also control whether an object writes to the depth buffer. If
``material.depth_write`` is False, objects behind it will still be drawn and visible (although the blending would be incorrect).

Objects that don't write depth are usually drawn after objects that do write depth.
In Pygfx, the default value of ``material.depth_write``
is True when ``alpha_method in ("opaque", "stochastic")`` or when ``alpha_mode="auto"``.


Not supported
-------------

Most render engines support the "opaque" and "blended" alpha methods. The
"weighted" and "stochastic" methods are generally conidered more special. But they can solve numerous use-cases,
and these methods have (more or less) the same performance as "opaque" or "blended".

There exist more advanced methods for dealing with transparency, such as dual
depth peeling, adaptive transparency, and a K-buffer. These methods can produce
very good results, but they suffer a significant penalty in terms of performance
and memory usage. This is why methods like these are currently not supported.


List of transparency use-cases
------------------------------

Here's a list of both common and special use-cases, explaining how to implement them in Pygfx, as well as in ThreeJs, for comparison.


* A fully opaque object

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "solid"

    .. code-block:: js

        // ThreeJS
        m.transparent = false;  // default

* Classic transparency (the over operator)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "blend"

    .. code-block:: js

        // ThreeJS
        m.transparent = true;
        m.depthWrite = false;

* Additive blending (glowy transparent objects)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "add"

    .. code-block:: js

        // ThreeJS
        m.transparent = true;
        m.blending = THREE.AdditiveBlending;
        m.depthWrite = False;

* Additive blending (glowy opaque objects)

    .. code-block:: py

        # Pygfx
        # (because depth_write is set, the render_queue will be 2600; smaller than 'real' transparent objects (3000))
        m.alpha_mode = "add"
        m.depth_write = True

    .. code-block:: js

        // ThreeJS
        // (configure to render the object at the end of the opaque pass)
        m.transparent = false;
        m.blending = THREE.AdditiveBlending;
        m.depthWrite = true;  // default
        ob.renderOrder = 99;

* Multiplicative blending (color tinting or darkening)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "multiply"

    .. code-block:: js

        // ThreeJS
        m.transparent = true;
        m.blending = THREE.MultiplyBlending;

* Custom blending

    .. code-block:: py

        # Pygfx
        m.alpha_config = {
            "method": "blended",
            "color_op": ..,  # wgpu.BlendOperation, default "add".
            "color_src": ..,  # wgpu.BlendFactor
            "color_dst": ..,  # wgpu.BlendFactor
            "color_constant": ..,  # default black
            "alpha_op": ..,  # wgpu.BlendOperation, default "add".
            "alpha_src": ..,  # wgpu.BlendFactor
            "alpha_dst": ..,  # wgpu.BlendFactor
            "alpha_constant": ..,  # default 0
        }

    .. code-block:: js

        // ThreeJS
        m.transparent = true;
        m.blending = THREE.CustomBlending;

        m.blendEquation = ..
        m.blendSrc = ..
        m.blendDst = ..
        m.blendColor = ..
        m.blendEquationAlpha = ..
        m.blendSrcAlpha = ..
        m.blendDstAlpha = ..
        m.blendAlpha = ..


* An opaque object with holes (a.k.a. alpha testing / masking)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "solid"
        m.alpha_test = 0.5

    .. code-block:: js

        // ThreeJS
        m.transparent = false;  // default
        m.alphaTest = 0.5;

* A transparent object with holes (alpha blending and testing)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "blend"
        m.alpha_test = 0.5

    .. code-block:: js

        // ThreeJS
        m.transparent = True;
        m.alphaTest = 0.5;

* A background

    .. code-block:: py

        # Pygfx
        ob.material.render_queue = 1000  # the render queue for backgrounds

    .. code-block:: js

        // ThreeJS
        // (put at the beginning of the opaque-pass)
        m.transparent = false;
        m.renderOrder = -99;

* An overlay

    .. code-block:: py

        # Pygfx
        ob.material.render_queue = 4000

    .. code-block:: js

        // ThreeJS
        // (put at the end of the transparency-pass, so no solid objects possible.)
        m.transparent = true;
        m.renderOrder = 99;

* Stochastic transparency

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "dither"

    .. code-block:: js

        // ThreeJS
        m.alphaHash = true;

* Order independent transparency

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "weighted_blend";

    .. code-block:: js

        // Not supported by the engine

