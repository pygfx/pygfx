Blending and transparency
=========================

When rendering objects with PyGfx, the shader calculates the color and alpha
value for each fragment (i.e. output pixel) and then writes these to the output
texture (the render target).

In the simplest case, the object is opaque, and the colors simply overwrites the
colors in the output texture. In other cases, the object are combined with
other objects, and there are multiple ways to do this.


Alpha
-----

Transparency is expressed using an ``alpha`` value. This can be the fourth
component in an RGBA color, e.g. in a color property or in a colormap. But it
can also be an explicit value, e.g. ``material.opacity``.

Although they are commonly used to represent transparency, this is not always
the case. E.g. they can be used as the reference value in alpha testing, or as a
weight in weighted blending.

In any case, the alpha value represents a weight for how the object is combined with
other objects, and it's applicance depends on ``material.alpha_mode``.


Alpha mode
----------

In Pygfx, ``material.alpha_mode`` determines how the alpha value of an object's fragment
is used to combine it with the output texture.

* "opaque": The fragment color is multiplied with alpha and then *overwrites* the value in the output texture.
* "blend": Per-fragment blending of the object's color and the color in the output texture.
* "dither": Stochastic transparency. The chance of a fragment being visible (i.e. not discarded) is alpha. Visible fragments are opaque.
* "weighted": Weighted blending, where the order of objects does not matter for the end-result. One use-case being order independent transparency (OIT).

When the mode is "blend", exta parameters can be passed to ``material.blend_mode``.
But you can also use the following presets, which set ``alpha_mode`` to "blend", and then
configure ``blend_mode`` appropriately:

* "over": use classic alpha blending using the *over operator*.
* "add": use additive blending that adds the fragment color, multiplied by alpha.
* "subtract": use subtractuve blending that removes the fragment color.
* "multiply": use multiplicative blending that multiplies the fragment color.

When the mode is "weighted", extra parameters can be passed to ``material.weighted_mode``.


Blending
--------

Blending means that objects are combined on a per-fragment basis. The object's
fragment color and the current color in the output texture are blended using a
mathematical formula. There are different kinds of blending, the most common
being the "over operator", which in Pygfx we call "over" blending.

When an object is blended, this has certain implications, the most obvious one being
that the result will depend on the order in which the objects are rendered.


Render order
------------

For objects that have their fragments blended, the order of rendering these
objects is important to get the correct result. The renderer sorts objects based
on three factors:

* Objects are first sorted by their ``object.render_order``. Users should generally not need this, but it allows full control over the render order.
* Then bases on their ``material.alpha mode`` and ``material.depth_write``, in order: "opaque" and "dither", "blend" with depth write, "blend", "weighted".
* The object's distance to the camera. Note that this won't help all cases, since objects
  can still intersect other objects (and themselves).

Even with these rules, object (or parts of objects) may be rendered "behind"
other objects, which uis usually not what we want. This is avoided with the depth buffer.


Depth buffer
------------

The depth buffer is a texture of the same size as the color output texture, that
stores the distance from the camera, of the last drawn fragment. If an object
has ``material.depth_test = True``, fragments that would be further from the
camera (i.e. are occluded by another object) will not be drawn. The ``material.depth_test`` is True by default.

The depth buffer also helps with performance of rendering opaque objects by
avoiding "overdraw". It does this by dropping fragments bases on their depth
value, without even calculating the fragment color. This optimization is known
as "early Z", and is why opaque objects are drawn front-to back. Note that it
only works when the depth of an object can be determined before running the
shader (i.e. the shader does not ``discard`` or set the depth). In Pygfx, meshes
support early-z, but lines, points and text do not.

One can also control whether an object writes to the depth buffer. If it does
not, it allows objects that are behind it to still be drawn (even though the
blending might not be theoretically correct).

Typically, objects that have opaque fragments are drawn first, with
``depth_write=True``, and transparent objects are drawn after, with
``depth_write=False``. In Pygfx, the default value of ``material.depth_write``
is True when ``alpha_mode`` is "opaque" or "dither", and False otherwise.



List of transparency use-cases
------------------------------

Here's a list of both common and special use-cases, explaining how to implement them in Pygfx, as well as in ThreeJs, for comparison.


* A fully opaque object

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "opaque"  # default if opacity == 1

    .. code-block:: js

        // ThreeJS
        m.transparent.false;  // default

* Classic transparency (the over operator)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "over"   # default if opacity < 1

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
        m.alpha_mode = "add"
        m.depth_write = True

    .. code-block:: js

        // ThreeJS
        m.transparent = false;
        m.blending = THREE.AdditiveBlending;
        m.depthWrite = true;  // default

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
        m.alpha_mode = "blend"

        m.blend_mode = {
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
        m.alpha_mode = "opaque"  # default if opacity == 1
        m.alpha_test = 0.5

    .. code-block:: js

        // ThreeJS
        m.transparent = false;  // default
        m.alphaTest = 0.5;

* A transparent object with holes (alpha blending and testing)

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "over"  # default if opacity < 1
        m.alpha_test = 0.5

    .. code-block:: js

        // ThreeJS
        m.transparent = True;
        m.alphaTest = 0.5;

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
        m.alpha_mode = "weighted";

    .. code-block:: js

        // Not supported by the engine
