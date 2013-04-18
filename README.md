DO++
==========

# Introduction
DO++ is a basic set of C++ libraries for computer vision.

I am aware OpenCV is out there and competing with it is out of question.
I simply invite you to give a try to DO++ because:
* DO++ has a very user-friendly API.
* DO++ is small and thus easily maintainable and extensible.
* DO++ is opensource and licensed with the MPL2 license.

# Currently available module
Currently added features are:
- Core module
- Graphics module
- Features module (incomplete)
- Image Processing module (incomplete)

The following modules are being cleaned and will be added in the process:
- FLANN Wrapper module
- Geometry module
- Feature Matching module
Please be patient...

A windows installer will be made available soon. Meanwhile you can generate it with CPack and NSIS.

# Third-Party Software
DO++ uses the following third-party libraries:

- [**Qt 5**](qt-project.org) is a cross-platform application and UI framework with APIs for C++ programming.
  Qt 5 is licensed with GNU Lesser General Public License (LGPL) version 2.1.
  
  Qt 5 is not included in the repository and must be installed. 
  
  If it is installed, then you must set system environment variable **QT5_DIR** which will be appended in **CMAKE_PREFIX_PATH** when generating projects with CMake. In addition, append the binary directory of Qt 5 (e.g., **%QT5_DIR%\bin** in windows environment) to the system **PATH**.
  
  I have made available compiled Qt5 Libraries for Visual Studio 2010 x64 and Visual Studio 2012 x64.
  They are available here:
  - [Qt 5.0.1 compiled with VS2010 64bit (ICU and WebKit disabled) (561 Mb)](https://dl.dropbox.com/u/80774144/repositories/qt-5.0.1-msvc2010-x64.7z)
  - [Qt 5.0.1 compiled with VS2012 64bit (ICU and WebKit disabled) (594 Mb)](https://dl.dropbox.com/u/80774144/repositories/qt-5.0.1-msvc2012-x64.7z)

- [**Eigen**](http://eigen.tuxfamily.org/) is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Eigen is licensed with Mozilla Public License v.2 (MPL2).
  
  **Eigen 3.1.2** is already included in the repository and it is therefore not needed to install it.

- [**Google Test**](https://code.google.com/p/googletest/) is Google's framework for writing C++ tests on a variety of platforms (Linux, Mac OS X, Windows, Cygwin, Windows CE, and Symbian). Based on the xUnit architecture. Supports automatic test discovery, a rich set of assertions, user-defined assertions, death tests, fatal and non-fatal failures, value- and type-parameterized tests, various options for running the tests, and XML test report generation. Google Test is licensed with New BSD License.
  
  Google Test 1.6.0 is already included in the repository and it is therefore not needed to install it.
