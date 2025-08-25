Anti-aliasing in PyGfx
======================

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

A HiDPI display (Apple calls it a retina display) typically has twice as many
pixels as a normal display; for every "logical pixel" there are four physical
pixels (2 in each dimension). In other words, the actual pixels are so small
that aliasing effects are much harder to see, and everything looks much smoother
and sharper.

That said, using one or more of the following techniques probably still adds to
the perceived quality. Especially if you have a large screen, because larger
screens have larger pixels.


Super-sample anti-aliasing (SSAA)
---------------------------------

PyGfx first renders the scene to an internal texture, and this texture can be a
different size than the region it will occupy on the screen. If the resolution
of the texture is larger, the image is filtered when it is drawn to the screen,
reducing the amount of aliasing.

This is known as super-sample aa or full-screen aa, and it is an effective and
straightforward technique. However, it is relatively costly in terms of memory
and performance (more pixels to render means more invocations for the fragment
shaders).

See ``renderer.pixel_ratio`` bla

In PyGfx the default is .... TODO ... TODO I want to revisit what pixel-ratio we
observe when peope use OS-level "scaling". Because if a user sets a high scale,
so logical pixels become large, the pixel-ratio should be large as well, I
guess. TODO What I mean is that instead of renderer.pixel_ratio, we may want
something relative to the monitor's native resolution, but then taking HiDPI
into account. TODO to turn it on ... TODO to turn it off ...


Post-processing anti-aliasing (PPAA)
------------------------------------

Post-processing anti-aliasing is a screen-space technique applied after the
image is rendered. It analyzes the final image and smooths jagged edges by
blurring or blending pixels along high-contrast areas. The downside is that it
is applied after the aliasing has already taken place, making it less effective
than e.g. SSAA in *preventing* aliasing. That said, it's a fast and lightweight
technique, it produces results that typically look smoother than with SSAA.


There exist a variety of PPAA algorithms, which vary in their speed,
effectiveness to reduce jaggies, and introduction of unintended blurring. A well
known algorithm is FXAA (fast approxomate aa). Pygfx by default uses DDAA
(direction diffusion aa). See ``renderer.ppaa`` to select the algorithm or turn it off.


Shader-based anti-aliasing
--------------------------


Some materials have a property ``material.aa``. When set to True, the shader
produces semi-transparent fragments to soften the object's edges. Objects that
support this include lines, points and text. The introduction of these
semi-transparent fragments results in extra smooth edges, but can also cause
artifacts if (parts of) objects are not blended in the right order. Note that
with alpha-methods "opaque" and "stochastic" the effect is not applied, because
these alpha-methods don't support per-fragment blending.


Not supported
-------------

Multisample anti-aliasing (MSAA), a common method intended mostly for mesh
objects, is currently not supported.


Recommendations
---------------

TODO
