# How to contribute examples

This folder contain the pygfx example scripts.


## Purpose

Examples can serve a few purposes:

* They demonstrate the capabilities of pygfx.
* They explain how a particular visualization is performed (i.e. documentation).
* They help make sure that pygfx does what it promises (i.e. testing).

When creating / modifying an example, keep in mind how they're used.

* People view them in the gallery docs (seeing a screenshot ot movie, but no interactivity).
* People run the code, can interact (use mouse/keyboard) and modify the code.
* The example is run in test, and when used as validation, the screenshot must be consistent.


## Testing

Examples have three possible modes with regards to testing. For this purpose the
`# example_testing:` comment must be present in each file.

* `# example_testing: no` - not tested.
* `# example_testing: run` - tested by running the code (including rendering a frame).
* `# example_testing: compare` - tested by running the code, rendering a frame, and comparing it to the reference (i.e. image comparisons).


## Gallery

There are a few modes by which an example can appear in the gallery.
For this purpose the `# sphinx_gallery_pygf` comment must be present in each
example.

* `sphinx_gallery_pygfx = 'hidden'` - not present in the gallery.
* `sphinx_gallery_pygfx = 'code'` - present only as code (don't run the code when building docs).
* `sphinx_gallery_pygfx = 'screenshot'` - present in the gallery with a screenshot.
* `sphinx_gallery_pygfx = 'animate 3s'` - present in the gallery with an animation (length specified in seconds).

Examples that are animated should ``from time import perf_counter``, and use it
to determine the speed of the animation. The ``perf_counter`` variable is
patched during rendering to simulate time. This way the animation is also
independent on e.g. movie fps.
