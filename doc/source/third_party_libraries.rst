Third-Party Libraries
=====================

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
  is licensed with `Mozilla Public License v.2 (MPL2) <http://www.mozilla.org/MPL/>`_:
  
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