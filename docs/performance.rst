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


Event handlers and memory
-------------------------

By default ``add_event_handler`` keeps a *strong* reference to the handler. If
an object registers one of its own bound methods on an object it owns (e.g. a
controller registering ``self._on_pointer_down`` on a scene it holds), this
may form a reference cycle if the event handler is a class::

    owner -> child -> owner._event_handlers -> bound method -> owner

Reference counting cannot break a cycle, so dropping your last reference to the
owner frees nothing until the *cyclic* garbage collector runs. That collector
is scheduled by allocation thresholds that count Python objects and are unaware
of GPU memory, so any textures and buffers the owner keeps alive can linger far
longer than expected -- long enough to exhaust VRAM in workflows that create
and discard many scenes.

Two tools help reclaim such resources promptly:

* Pass ``weak=True`` to ``add_event_handler``. The handler is then stored via a
  weak reference, so the registration no longer keeps the callback (or, for a
  bound method, its instance) alive, and the owner is reclaimed by reference
  counting alone. Keep a reference to the callback yourself, otherwise it is
  collected and never fires (see :meth:`~pygfx.objects.EventTarget.add_event_handler`).
* Call :meth:`~pygfx.objects.EventTarget.clear_event_handlers` to deterministically
  detach all handlers from an object before discarding it.


Probably more
-------------

This document is a work in progress. Let us know if we forgot something!
