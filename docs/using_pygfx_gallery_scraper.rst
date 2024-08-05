Using the Pygfx gallery scraper
===============================

Pygfx implements its own scraper for Sphinx-gallery. Upstream projects that want
to create a gallery using Pygfx, can make use of it.


In conf.py
----------

Setup sphinx-gallery as usual. See the Sphinx-gallery docs.

When creating the conf, do the following:

.. code-block:: py

    sphinx_gallery_conf = {
        ...
        "image_scrapers": ("pygfx",),
    }

    from pygfx.utils.gallery_scraper import find_examples_for_gallery

    extra_conf = find_examples_for_gallery(your_examples_dir)
    sphinx_gallery_conf.update(extra_conf)

That last line sets the "examples_dirs", "ignore_pattern", and "filename_pattern" of the config dict.


Writing examples
----------------

There are a few modes by which an example can appear in the gallery.
For this purpose the ``# sphinx_gallery_pygfx_docs`` comment must be present in each
example that you want to have in the docs.

* ``sphinx_gallery_pygfx_docs = 'hidden'`` - not present in the gallery.
* ``sphinx_gallery_pygfx_docs = 'code'`` - present only as code (don't run the code when building docs).
* ``sphinx_gallery_pygfx_docs = 'screenshot'`` - present in the gallery with a screenshot.
* ``sphinx_gallery_pygfx_docs = 'animate 3s'`` - present in the gallery with an animation (length specified in seconds).


Finding the canvas
------------------

The Pygfx scraper needs to find the canvas or renderer to take the screenshot from.
It does this by looking for an object called ``disp``, ``renderer`` or ``canvas``.

If your library hides these details, we're open to expanding the logic so it is
able to detect the canvas in more use-cases.

As a workaround, you can always write something like ``canvas = your_object.xxx.yyy.canvas``.


Alternative usage
-----------------

You don't have to use the ``find_examples_for_gallery()`` to collect examples.
You can also set "ignore_pattern" and "filename_pattern" yourself. However, if
the scaper does not see the comment, it won't render a screenshot.

You can also create your own scraper and re-use some parts of the Pygfx scraper.
We tried to write ```pygfx/utils/gallery_scraper.py`` such that the code is very
much re-usable.
