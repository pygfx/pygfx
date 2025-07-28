Transparency
============

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
other objects, and it's applicance is fully defined by ``material.alpha_config``.
Although this config can be set in great detail, the majority of cases can be
captured with a handful of presets that we call *alpha modes*.


Alpha modes
-----------

In Pygfx, ``material.alpha_mode`` can be used to define how the alpha value of an object's fragment
is used to combine it with the output texture. There are a range of values to choose from, divided over four different methods:

Method "opaque" (overwrites the value in the output texture):

* "solid": alpha is ignored.
* "solid_premul": the alpha is multipled with the color (making it darker).

Method "stochastic" (alpha represents the chance of a fragment being visible):

* "dither": stochastic transparency with blue noise.
* "bayer": stochastic transparency with a Bayer pattern.

Method "composite" (per-fragment blending of the object's color and the color in the output texture):

* "blend": use classic alpha blending using the over-operator.
* "add": use additive blending that adds the fragment color, multiplied by alpha.
* "subtract": use subtractuve blending that removes the fragment color.
* "multiply": use multiplicative blending that multiplies the fragment color.

Method "weighted" (order independent blending):

* "weighted_blend": weighted blended order independent transparency.
* "weighted_solid": fragments are combined based on alpha, but the final alpha is always 1. Great for e.g. image stitching.

And finally, there is the "auto" mode, which is the same as "blend", except that when used, the ``depth_write`` defaults to
True if ``opacity==1``. This gives sensible results out of the box for a relatively wide range of use-cases. When artifacts
occur, consider any of the other modes.


Alpha methods
-------------

Most users don't have to worry much about what the methods mean. Though it's good to understand
that the "opaque" and "stochastic" methods produce opaque fragments, and by default have ``depth_write=True``.
The renderer sorts these objects front-to back to avoid overdraw (for performance).

In contrast, the "composite" and "weighted" methods result in semi-transparent fragments,
and by default have ``depth_write=False``. The renderer sorts these object back to front to
improve the chance of correct blending. Note that the 'auto' mode is an exception to this rule.


Alpha methods in detail
-----------------------

**Alpha method 'opaque'** represents no transparency. The fragment color
overwrites the value in the output texture. A very common method in render engines.

**Alpha method 'composite'** represents alpha compositing: a common method in render
engines in which objects are combined on a per-fragment basis. The object's
fragment color and the current color in the output texture are blended using a
mathematical formula. There are several common compositing configurations, the
most-used being the "over operator" (also known as normal blending). When alpha
compositing is used, the result will depend on the order in which the objects are
rendered.

**Alpha method 'stochastic'** represents stochastic transparency. The alpha
represents the chance of a fragment being visible (i.e. not discarded). Visible
fragments are opaque. This blend method is less common, but has interesting properties.
Although the result has a somewhat noisy appearance, it handles transparency perfectly,
capable of rendering multiple layers of transparent objects, and correctly handling
objects that have a mix of opaque and transparent fragments.

**Alpha method 'weighted'** represents (variants of) weighted blended order
independent transparency. The order of objects does not matter for the
end-result. One use-case being order independent transparency (OIT).
The order-independent property is advantageous in some use-cases, but produces
unfavourable results in others. It's use extends beyond transparency though, and
can also be used for e.g. image stiching.


Alpha config
------------

The ``material.alpha_config`` is a dictionary that fully describes how the combining based on alpha occurs.
This dictionary has at least three keys: the 'method', 'mode', and 'pass'. It may have additional keys for the options
available for the used method. The different presets represent common combinations of these options.

Most users just set ``material.alpha_mode`` which implicitly sets
``material.alpha_config``. In advanced/special cases, users can set the
``material.alpha_config`` directly to take full control over all available
options. In this case the 'mode' field and ``material.alpha_mode`` become "custom".


Render group and render order
-----------------------------

The ``object.render_order`` is a float that allows users to more precisely
control the order in which objects are rendered with respect to other objects in
the same pass (the opaque or transparent render pass). You typically don't need
this, but when you do, it's good that you can. The value applies to the object
and its children.

The ``object.render_group`` is a signed integer that enables grouping of
objects. Marking an object with ``.render_group=1`` means that the object (and
its children) are rendered later, as an overlay. Similarly a negative value can
be used for backgrounds. Note that this behaviour can always be implemented
using multiple calls to ``renderer.render()``, but allowing to specify multiple
render groups in one scene has its advantages.


How the renderer sorts objects
------------------------------

The renderer sorts objects based on the following factors:

* The ``object.render_group``.
* The ``material.alpha_config['pass']`` TODO: what's the final name of this field?
* The ``object.render_order``.
* The object's distance to the camera. The order is front to back in the
  opaque-pass, and back-to-front in the transparency-pass (to get correct
  blending). Weighted objects are not sorted.

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
is True when ``material.alpha_method`` is "opaque" or "stochastic", and False otherwise.


List of transparency use-cases
------------------------------

Here's a list of both common and special use-cases, explaining how to implement them in Pygfx, as well as in ThreeJs, for comparison.


* A fully opaque object

    .. code-block:: py

        # Pygfx
        m.alpha_mode = "solid"

    .. code-block:: js

        // ThreeJS
        m.transparent.false;  // default

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
        # (the object gets automatically rendered at the very start of the transparency-pass)
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
            "method": "composite",
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
        ob.render_group = -1

    .. code-block:: js

        // ThreeJS
        // (put at the beginning of the opaque-pass, so no blending possible.)
        m.transparent = false;
        m.renderOrder = -99;

* An overlay

    .. code-block:: py

        # Pygfx
        ob.render_group = 1

    .. code-block:: js

        // ThreeJS
        // (put at the end of the transparenct-pass, so no solid objects possible.)
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

