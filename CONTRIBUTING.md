# Contributor's Guide

## Who can contribute?

We accept contributions from anyone as long as these contributions meet our standards.
While we will accept contributions from anyone, we especially value ideas and
contributions from folks with diverse backgrounds and identities. There are
many ways to contribute (see below) and no contribution is too small.

## What can be contributed?

At this time, the project is still in a somewhat experimental fase of development.
Some parts of the architecture are not settled yet, and several decisions about
the API are yet to be made. Further the API is not complete yet.

1. **Try the examples**: in any case, the examples should work. Try them out and
   report a big if one does not work. This helps us improve the stability of Pygfx.
1. **Play with it**: If you're up for it, try to create your own visualizations
   with Pygfx, and let us know if you run into problems or if you have suggestions
   about the API. This will help us set the direction of the API.
4. **Documentation**: We could use help to improve the documentations, especially
    tutorials.


## How can I contribute?

Almost all communication should be done through
the main GitHub repository: https://github.com/pygfx/pygfx

* Bug reports and feature requests can be submitted through the "Issues" on
  the repository.
  [This GitHub page](https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/creating-an-issue)
  can help you create an issue if you're unfamiliar with the process.
* Any changes to actual code, including documentation, should be submitted
  as a pull request on GitHub. GitHub's documentation includes instructions
  on [making a pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)
  if you're new to GitHub or to git in general. Please make sure to submit
  pull requests using a **new** branch (not your fork's main branch).
  Don't be afraid to submit a pull request with only partial fixes/features.
  Pull requests can always be updated after they are created. Creating them
  early gives maintainers a chance to provide an early review of your code if
  that's something you're looking for.

No matter how you contribute, the maintainers will try their best to read,
research, and respond to your query as soon as possible. For code changes,
automated checks and tests will run on GitHub to provide an initial "review"
of your changes.

## What if I need help?

Currently, the best way to ask for help from the maintainers is to start a
[discussion](https://github.com/pygfx/pygfx/discussions).


## Coding Style

The pygfx project uses `black` to automatically format the code. Additionally
`flake8` is used to ensure "clean code". Use these commands:

```bash
# Run black to auto-format all code in the current directory
black .
# Run flake8 to check the code quality
flake8 .
```


## Logging

You can set the `PYGFX_LOG_LEVEL` environment variable to get more
detailed log messages. Can be an int or any of the standard level names.


## Attributes

We distinguish four kinds of attributes:

* Public attibutes for the user: should in most cases be ``@property``'s.
* Private attributes: prefixed with "_" as usual.
* Attributes used by other parts of pygfx but not intended for the user: prefixed with "_gfx_".
* Attributes to store wgpu-specific data on a WorldObject or Material (the objects themselves are unaware): prefixed with "_wgpu_".
