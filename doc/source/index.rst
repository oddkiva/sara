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
   reference/core
   :maxdepth: 2


Introduction
============

DO-CV is a basic set of C++ libraries for computer vision.

- DO-CV has a very user-friendly API.
- DO-CV is small, easily maintenable and extensible.
- DO-CV is opensource and licensed with the MPL2 license.
  
Try DO-CV by getting it at: https://github.com/davidok8/doplusplus.
  
Currently Available Modules
---------------------------

Currently added features are:

- *Core* module
- *Graphics* module
- *Image Processing* module (incomplete)
- *Features* module (incomplete)

The following modules are being cleaned and will be added in the process:

- *FLANN Wrappers* module
- *Geometry* module
- *Feature Matching* module

Please be patient...

A windows installer will be made available soon. Meanwhile you can generate it
with CPack and NSIS.

Third-Party Libraries
---------------------
DO-CV uses the following third-party libraries:

- `Qt 5`_ is a cross-platform application and UI framework with APIs for C++
  programming.  `Qt 5`_ is licensed with `GNU Lesser General Public License
  (LGPL) version 2.1
  <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.

  `Qt 5`_ is not included in the repository and **must be installed**.  If it is
  installed, then you must set system environment variable **QT5_DIR** which
  will be appended in **CMAKE_PREFIX_PATH** when generating projects with
  CMake. In addition, append the binary directory of `Qt 5`_ (e.g.,
  `%QT5_DIR\%\\bin` in windows environment) to the system `PATH`.

  .. _`Qt 5`: http://qt-project.org/
  
- Eigen_ is a C++ template library for linear
  algebra: matrices, vectors, numerical solvers, and related algorithms. Eigen
  is licensed with _`Mozilla Public License v.2 (MPL2)
  <http://www.mozilla.org/MPL/>`:
  
  **Eigen 3.2.0** is already included in the repository and it is therefore
  not needed to install it.

  .. _Eigen: http://eigen.tuxfamily.org/

- `Google Test`_ is Google's framework
  for writing C++ tests on a variety of platforms (Linux, Mac OS X, Windows,
  Cygwin, Windows CE, and Symbian). Based on the xUnit architecture. Supports
  automatic test discovery, a rich set of assertions, user-defined assertions,
  death tests, fatal and non-fatal failures, value- and type-parameterized
  tests, various options for running the tests, and XML test report generation.
  `Google Test`_ is licensed with New BSD License.
  
  **Google Test 1.6.0** is already included in the repository and it is
  therefore not needed to install it.

  .. _`Google Test`: https://code.google.com/p/googletest/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
