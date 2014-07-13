.. DO-CV documentation master file, created by
   sphinx-quickstart on Thu Jul 03 01:18:59 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DO-CV's documentation!
=================================

DO-CV is a small and easy-to-use C++ computer vision library.

DO-CV focuses on:

1. having an **easy-to-use and simple API**,
2. having **human-readable and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.

DO-CV is licensed with the `Mozilla Public License version
2.0 <https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2>`_.


Build the libraries
-------------------

To build the libraries, run::

  sudo apt-get install cmake
  sudo apt-get install -qq qtbase5-dev

  mkdir build
  cd build
  cmake ..
  make  -j N  # N is the number of CPU cores you want to use.

  make test


Table of Contents
-----------------
.. toctree::
   :maxdepth: 2

   ref_doc_toc
   examples
   third_party_libraries
   

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
