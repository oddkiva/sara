{{ section }}

.. toctree::
   :maxdepth: 2
{% for module in modules %}
   {{ module }}
{%- endfor %}


.. doxygenfile:: {{ master_header }}
