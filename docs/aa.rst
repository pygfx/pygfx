----------------------
Anti-aliasing in PyGfx
----------------------

Aliasing in a render engine is the visual distortion that happens when
high-frequency detail (like thin lines, edges, or patterns) is sampled at too
low a resolution. It usually shows up as jagged edges (“jaggies”), flickering,
or Moiré patterns. This happens because the discrete pixels can't capture the
smooth transitions of the original image.

Anti-aliasing techniques reduce this effect. PyGfx support the following aa
(anti-aliasing) techniques:


Your monitor
------------

Ok, this is not a technique provided by PyGfx, but let's discuss it first
because it matters, and it affects to what degree you need the other techniques.

A HiDPI display (Apple calls it a Retina display) typically has twice as many
pixels as a normal display; for every "logical pixel" there are four physical
pixels (2 in each dimension). In other words, the actual pixels are so small
that aliasing effects are much less pronounced.


Super-sample anti-aliasing (SSAA)
---------------------------------

PyGfx first renders the scene to an internal texture, and this texture can be a
different size than the region it will occupy on the screen. If the resolution
of the texture is larger, the image is filtered when it is drawn to the screen,
reducing the amount of aliasing.

This is known as SSAA (super-sample anti-aliasing), and it is an effective and
straightforward technique. However, it is relatively costly in terms of memory
and performance (more pixels to render means more invocations for the fragment
shaders).

In PyGfx, the ``renderer.pixel_scale`` determines the scale in resolution of the internal texture:
a value > 1 means that SSAA is used. A value < 1 means that a lower-resolution internal texture
is used, which is then upscaled to the screen resolution (a performance trick).

By default, ``pixel_scale`` is 2 for normal displays, and 1 for HiDPI
displays. In effect, the resolution of the internal texture is more or less
independent of the screen being used. If you don't want SSAA, set
``renderer.pixel_scale=1``.

The ``renderer.pixel_filter`` determines the filter being used. The default is the
Mitchell filter, a quadratic filter suitable for image reconstruction.
When ``pixel_scale==1``, the texture image is simply copied without a filter.


Post-processing anti-aliasing (PPAA)
------------------------------------

Post-processing anti-aliasing is a screen-space technique applied after the
image is rendered. It analyzes the final image and smooths jagged edges by
blurring or blending pixels along high-contrast areas. The downside is that it
is applied *after* the aliasing has already taken place, making it less effective
than e.g. SSAA in *preventing* aliasing. That said, it's a fast and lightweight
technique, and produces results that typically look smoother than with (only) SSAA.

There exist a variety of PPAA algorithms, which vary in their speed,
effectiveness to reduce jaggies, and introduction of unintended blurring. A well
known algorithm is FXAA (fast approxomate anti-aliasing). Pygfx by default uses DDAA
(directional-diffusion anti-aliasing), which has improved performance and image quality compared to FXAA.
See ``renderer.ppaa`` to select the algorithm or turn it off.


Shader-based anti-aliasing
--------------------------

Some materials have a property ``material.aa``. When set to True, the shader
produces semi-transparent fragments to soften the object's edges. Objects that
support this include lines, points and text. The introduction of these
semi-transparent fragments results in extra smooth edges, but can also cause
artifacts if (parts of) objects are not blended in the right order.

The ``material.aa`` property is False by default. We generally recommend setting
it to True for text objects, because it improves the quality of the rendered
text.

Note that with alpha-methods "opaque" and "stochastic" the effect is not
applied, because these alpha-methods don't support per-fragment blending.


Not supported
-------------

Multi-sample anti-aliasing (MSAA), a common method intended mostly for mesh
objects, is currently not supported.


Summary
-------

PyGfx by default uses a combination of SSAA (if not on a HiDPI monitor), and PPAA.
Shader-based anti-aliasing is supported for lines, points, and text. It's off
by default to prevent blending artifacts, but generally recommended to turn on for text.
