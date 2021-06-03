import jinja2


jinja_env = jinja2.Environment(
    block_start_string="{$",
    block_end_string="$}",
    variable_start_string="{{",
    variable_end_string="}}",
    line_statement_prefix="$$",
    undefined=jinja2.StrictUndefined,
)


class BaseShader:
    """Base shader object to compose and template shaders using jinja2."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __getitem__(self, key):
        return self.kwargs[key]

    def get_code(self):
        raise NotImplementedError()

    def get_final_code(self, **kwargs):
        code = self.get_code()
        t = jinja_env.from_string(code)

        variables = self.kwargs.copy()
        variables.update(kwargs)

        try:
            return t.render(**variables)
        except jinja2.UndefinedError as err:
            msg = f"Canot compose shader: {err.message}"
        raise ValueError(msg)  # don't raise within handler to avoid recursive tb
