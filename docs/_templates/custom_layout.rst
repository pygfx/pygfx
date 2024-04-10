{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}


{% if objtype == "module" and name in [] %}
{# Some modules look nicer as a single-page doc. #}

.. automodule:: {{ fullname }}
  :members:

{% elif objtype == "module" %}

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
