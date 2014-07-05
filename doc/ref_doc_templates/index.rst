.. DO-CV documentation master file, created by
   sphinx-quickstart on Thu Jul 03 01:18:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DO-CV's documentation!
=================================

This is still experimental.

Reference Documentation
-----------------------

List of modules:

.. toctree::
{%- for library in libraries %}
   reference/{{ library }}
{%- endfor %}
   :maxdepth: 1

.. include:: introduction


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
