# How to contribute examples

This folder contain the Pygfx example scripts.


## Purpose

Examples can serve a few purposes:

* They demonstrate the capabilities of Pygfx.
* They explain how a particular visualization is performed (i.e. documentation).
* They help make sure that Pygfx does what it promises (i.e. testing).

When creating / modifying an example, keep in mind how they're used.

* People view them in the gallery docs (seeing a screenshot ot movie, but no interactivity).
* People run the code, can interact (use mouse/keyboard) and modify the code.
* The example is run in test, and when used as validation, the screenshot must be consistent.


## Testing

Examples have three possible modes with regards to testing. For this purpose the
`# sphinx_gallery_pygfx_test` comment must be present in each example. Note that
the tests have nothing to do with sphinx-gallery, but naming it as such prevents
these comments from being shown in the gallery.

* `# sphinx_gallery_pygfx_test = 'off'` - not tested.
* `# sphinx_gallery_pygfx_test = 'run'` - tested by running the code (including rendering a frame).
* `# sphinx_gallery_pygfx_test = 'compare'` - tested by running the code, rendering a frame, and comparing it to the reference (i.e. image comparisons).


While the images can be generated locally, they often result in slight
differences from the ones generated on the CI. For now, we recommend:

1. Ensure the example works well locally.
2. Get the example to run on the GitHub Actions CI job nammed `Screenshots / Regenerate`.
3. Download the artifacts.
4. Ensure they are of adequate quality.
5. Add or update the relevant screenshots.
6. Push the changes.

## Gallery

There are a few modes by which an example can appear in the gallery.
For this purpose the `# sphinx_gallery_pygfx_docs` comment must be present in each
example.

* `sphinx_gallery_pygfx_docs = 'hidden'` - not present in the gallery.
* `sphinx_gallery_pygfx_docs = 'code'` - present only as code (don't run the code when building docs).
* `sphinx_gallery_pygfx_docs = 'screenshot'` - present in the gallery with a screenshot.
* `sphinx_gallery_pygfx_docs = 'animate 3s'` - present in the gallery with an animation (length specified in seconds).

Examples that are animated should ``from time import perf_counter``, and use it
to determine the speed of the animation. The ``perf_counter`` variable is
patched during rendering to simulate time. This way the animation is also
independent on e.g. movie fps.
