{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == "function" %}
.. autofunction:: {{ objname }}
{% elif objtype == "module" %}
.. automodule:: {{ fullname }}
{% else %}
.. autoclass:: {{ objname }}
   {% block methods %}
   {% if methods %}
   
   {# work out if there are local methods #}
   {# Note: jinja hack to work around missing list comprehension #}
   {% set local_methods = [] %}
   {% for item in methods %}
      {% if not item.startswith("__") %}
      {% if item not in inherited_members %}
      {% set foo = local_methods.append(item) %}
      {% endif %}
      {% endif %}
   {% endfor %}
   {% endif %}
   
   {% if local_methods %}
   .. rubric:: {{ _('Methods') }}
   .. autosummary::

   {% for item in local_methods %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   {# work out if there are local attributes #}
   {% set local_attributes = [] %}
   {% for item in attributes %}
      {% if item not in inherited_members %}
      {% set foo = local_attributes.append(item) %}
      {% endif %}
   {% endfor %}
   {% endif %}
   
   {% if local_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::

   {% for item in local_attributes %}
      ~{{ name }}.{{ item }}
   {% endfor %}
   {% endif %}
   {% endblock %}

{% endif %}
{% if objtype != "module" %}
   .. rubric:: Examples
   .. minigallery:: pygfx.{{name}}
{% endif %}