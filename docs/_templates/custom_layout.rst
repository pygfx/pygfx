{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}


{% if objtype == "module" %}
{# Assume that the module docstring has an autosummary section to produce a list of members. #}

.. automodule:: {{ fullname }}

{% elif objtype == "function" %}

.. autofunction:: {{ objname }}

.. minigallery:: pygfx.{{ objname }}
    :add-heading: Examples
    :heading-level: ^

{% elif objtype == "class" %}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :member-order: bysource

.. minigallery:: pygfx.{{ objname }}
    :add-heading: Examples

{% else %}

.. autodata:: {{ objname }}

{% endif %}
