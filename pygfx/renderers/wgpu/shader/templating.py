import jinja2

root_loader = jinja2.PrefixLoader({}, delimiter=".")

jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
    loader=root_loader,
)


def register_wgsl_loader(context, loader):
    """Register a source for shader snippets.

    When code is encountered that looks like::

       {$ include 'some_context.name.wgsl' $}

    The loader for "some_context" is looked up and used to load the wgsl to include.
    This function allows registering a loader for your downstream package or application.

    Parameters
    ----------
    context : str
        The context of the loader.
    loader: jinja2.BaseLoader | callable | dict
        The loader to use for this context. If a function is given, it must accept one
        positional argument (the name to include).
    """
    if not (isinstance(context, str) and "." not in context):
        raise TypeError("Wgsl load context must be a string witout dots.")
    if context in root_loader.mapping:
        raise RuntimeError(f"A loader is already registered for '{context}'.")
    if isinstance(loader, jinja2.BaseLoader):
        root_loader.mapping[context] = loader
    elif isinstance(loader, dict):
        root_loader.mapping[context] = jinja2.DictLoader(loader)
    elif callable(loader):
        root_loader.mapping[context] = jinja2.FunctionLoader(loader)
    else:
        raise TypeError(
            f"The given wgsl loader must be a jinja2.BaseLoader, function, or dict. Not {loader!r}"
        )


register_wgsl_loader("pygfx", jinja2.PackageLoader("pygfx.renderers.wgpu.wgsl", "."))


def apply_templating(code, **kwargs):
    t = jinja_env.from_string(code)
    try:
        return t.render(**kwargs)
    except jinja2.UndefinedError as err:
        raise ValueError(f"Cannot compose shader: {err.args[0]}") from None
