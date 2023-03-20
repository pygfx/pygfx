{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == "function" %}
.. autofunction:: {{ objname }}
{% elif objtype == "module" %}
.. automodule:: {{ fullname }}
{% else %}
.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :member-order: bysource

{% endif %}
{% if objtype != "module" %}
   .. rubric:: Examples
   .. minigallery:: pygfx.{{name}}
{% endif %}