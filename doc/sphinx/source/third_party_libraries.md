Third-Party Libraries
=====================

Sara uses the following third-party libraries:

- `Qt 5`_ is a cross-platform application and UI framework with APIs for C++
  programming.  `Qt 5`_ is licensed with `GNU Lesser General Public License
  (LGPL) version 2.1
  <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.

  `Qt 5`_ is not included in the repository and **must be installed**.  If it is
  installed, then you must set system environment variable **QT5_DIR** which
  will be appended in **CMAKE_PREFIX_PATH** when generating projects with CMake.
  In addition, append the binary directory of `Qt 5`_ (e.g., `%QT5_DIR\%\\bin`
  in windows environment) to the system `PATH`.

  .. _`Qt 5`: http://qt-project.org/

- Eigen_ is a C++ template library for linear
  algebra: matrices, vectors, numerical solvers, and related algorithms. Eigen
  is licensed with `Mozilla Public License v.2 (MPL2)
  <http://www.mozilla.org/MPL/>`_:

  Eigen_ is already included in the repository and it is therefore
  not needed to install it.

  .. _Eigen: http://eigen.tuxfamily.org/

- `Boost`_ libraries.

  .. _Boost: https://www.boost.org
