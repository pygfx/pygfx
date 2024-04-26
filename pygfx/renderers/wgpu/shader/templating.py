import jinja2

loader = jinja2.PrefixLoader({}, delimiter=".")
loader.mapping["pygfx"] = jinja2.PackageLoader("pygfx.renderers.wgpu.wgsl", ".")

jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
    loader=loader,
)


def register_wgsl_loader(context, func):
    """Register a source for shader snippets.

    When code is encountered that looks like::

       {$ include 'some_context.name.wgsl' $}

    The loader for "some_context" is looked up and used to load the wgsl to include.
    This function allows registering a loader for your downstream package.

    Parameters
    ----------
    context : str
        The context of the loader.
    func: callable
        The function that will be called when a shader is loaded for the given context.
        The function must accept one positional argument (the name to include).
    """
    if not (isinstance(context, str) and "." not in context):
        raise TypeError("Wgsl load context must be a string witout dots.")
    if not callable(func):
        raise TypeError("The given wgsl load func must be callable.")
    if context in loader.mapping:
        raise RuntimeError(f"A loader is already registered for '{context}'.")
    loader.mapping[context] = func


def apply_templating(code, **kwargs):
    t = jinja_env.from_string(code)
    try:
        return t.render(**kwargs)
    except jinja2.UndefinedError as err:
        raise ValueError(f"Cannot compose shader: {err.args[0]}") from None
