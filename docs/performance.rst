--------------------
Performance in PyGfx
--------------------

PyGfx has numerous options to balance performance with quality. We try to select
defaults that are sensible for the majority of use-cases, probably leaning a bit towards quality.
But each use-case is different.

In this document we give an overview of what knobs you can turn to increase performance (or quality).


Pixel-scale, pixel-ratio, and upsampling
----------------------------------------

The value of ``render.pixel_scale`` is 1 for hiDPI screens and 2 otherwise. A value larger than one means that SSAA (super-sample anti-aliasing) is used, which
costs affects memory usage and performance. You can set ``renderer.pixel_scale=1`` to disable SSAA.

You can also set ``renderer.pixel_scale`` or ``renderer.pixel_ratio`` to values smaller than one, so the image is rendered to
a smaller internal texture, which is then upsampled to the screen resolution.

When you're setting ``pixel_scale`` to values ``>1``, keep in mind that the pixel filter (see ``renderer.pixel_filter``) is optimized for ``pixel_scale==2``.


Other anti-aliasing
-------------------

The ``rendere.ppaa`` controls the PPAA (post-procesing anti-aliasing). It defaults to "ddaa" to improve the visual result of the rendered image.
You can set ``renderer.ppaa='none'`` to disable it.

The ``material.aa`` available on text, lines, and points should not significantly affect performance. This means that, if your scene
does not contain intersecting objects, you can turn it on, turn off the PPAA or SSAA, and still have comparable quality.


Opaque objects
--------------

The default ``alpha_mode='auto'`` handles both opaque and transpareny objects. If you know that your object is opaque (i.e. solid), set ``material.alpha_mode='solid'``.
The renderer sorts opaque objects front-to-back to avoid overdraw, which improves performance.


Picking and clipping
--------------------

To enable picking on an object, you must set ``material.pick_write=True``.
Although it should not affect performance a lot, it may become significant if
applied to a lot of objects. So keep in mind to only turn it on when needed.

Same for ``material.clipping_planes``.


Shadows and lights
------------------

The shader iterates over each light to do lighting calculations. A handfull of lights should be fine, but don't go crazy.
The renderer must create a shadow map for every light in the scene. Take that in mind when adding lights and setting ``material.cast_shadow``.


Canvas update-mode
------------------

The canvas ``update_mode`` that can be provided whan a canvas is created, and set with ``canvas.set_update_mode()`` can be set to "manual", "ondemand", "continuous", and "fastest".
The ondemand mode is intended to only draw when there is a change, saving your laptop battery. The continuous mode keeps drawing, at a preset maximum FPS. If you wanna go as fast
as your system can, use "fastest". It may then still stick to your system's vsync, which can be turned off when creating a canvas (``RenderCanvas(..., vsync=False)``).


Data updates
------------

Whenever the data of a texture or buffer is changed, it needs to be uploaded to the GPU before the next draw.
If this is a lot of data (large data, or many small buffers/textures) this can affect performance.


Object transforms
-----------------

The logic behind object transform has been optimied *a lot*, but it's all Python
and Numpy. So in use-cases with many changes to object transformations in deep
nested scene graphs can affect performance.


Events
------

Doing a lot of work on certain events can affect performance. Especially events like "pointer_move".
Some seemingly hamless things, like using ``print()`` on every draw, are more expensive than you might think.


Probably more
-------------

This document is a work in progress. Let us know if we forgot something!
