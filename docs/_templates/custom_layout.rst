{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{% if objtype == "module" %}

.. automodule:: {{ fullname }}

{% elif objtype == "function" %}

.. autofunction:: {{ objname }}

.. minigallery:: pygfx.{{ objname }}
    :add-heading: Examples
    :heading-level: ^

{% else %}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :member-order: bysource

.. minigallery:: pygfx.{{ objname }}
    :add-heading: Examples

{% endif %}
