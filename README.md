=====
DO-CV
=====

------------
Introduction
------------

DO-CV is a basic set of C++ libraries for computer vision.

I am aware OpenCV is out there and competing with it is out of question.
Still, I invite to give it a try:
* DO-CV has a very user-friendly API
* DO-CV is small and thus easily maintainable and extensible.

-------
License
-------

See [LICENSE MPL2](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2) license.


-------
Authors
-------

See [Authors](https://github.com/openMVG/openMVG/raw/master/AUTHORS) text file.


--------
Building
--------

Continuous integration using [Travis CI](www.travis-ci.org) (master branch, linux 64 bits, BUILD + UNIT TESTING):

[![Build Status](https://travis-ci.org/DO-CV/DO-CV.png?branch=master)](https://travis-ci.org/DO-CV/DO-CV)

-------------
Documentation
-------------

Documentation is still under construction. Have a look at the doxygen-generated documentation [here](http://do-cv.github.io/DO-CV/).

Currently, the best way to see how the library works is to look at samples in the **test** folder.

---------------------------
Currently Available Modules
---------------------------
Currently added features are:
- **Core** module
  Clean C++ template data structures:
  * Eigen-based linear algebra integration and extension.
  * N-dimensional array and N-dimensional iterators
  * Image classes and color classes, color conversion and color rescaling.
  * High resolution timer
  * Tree data structure
- **Graphics** module
  * Cross-platform
  * Easy-to-use API for quick-prototyping
  (has less features than in OpenCV but more convenient to use than in OpenCV)
- **Image Processing** module
  * Basic linear filtering (Sobel, Kirsch, Roberts-Cross, Gaussian blur...)
  * Interpolation
  * Image warping
  * Deriche IIR filters
  * Differential calculus, norm, orientation, second moment matrix...
  * Image pyramid data structure, Gaussian pyramid, DoG, LoG
  * Image processing filters can be chained in a very few lines and easy to understand.
  * Reduce, enlarge, upscale, downscale images.
- **Features** module
  * Feature and descriptor data structures
  * I/O compatible with Bundler format.
- **Feature Detectors** module
  * Refinement/interpolation of keypoint localization based on Newton's method.
  * DoG extrema
  * scale normalized LoG extrema
  * Harris-Laplace corners
  * multiscale determinant of Hessian extrema
  * Affine shape adaptation based on the second moment-matrix
  * Scale selection based on the normalized Laplacian of Gaussians
  * Dominant gradient orientations of patch
  * SIFT implementation
  * Adaptive non-maximal suppression
  * Hessian-Laplace extrema (coming very soon, implementation straightforward, indeed see test_Hessian.cpp for now).

The following modules are being cleaned and will be added in the process:
- **FLANN Wrapper** module
- **Geometry** module (in the repository but need clearning)
- **Feature Matching** module (need cleaning)

Please be patient...

A windows installer will be made available soon. Meanwhile you can generate it with CPack and NSIS.


--------------------
Third-Party Software
--------------------

DO++ uses the following third-party libraries:

- [**Qt 5**](qt-project.org) is a cross-platform application and UI framework with APIs for C++ programming.
  Qt 5 is licensed with GNU Lesser General Public License (LGPL) version 2.1.
  
  Qt 5 is not included in the repository and must be installed. 
  
  *The following paragraph might be applicable only for Windows and Mac OS X platform:* if Qt 5 is installed, then you must set system environment variable **QT5_DIR** which will be appended in **CMAKE_PREFIX_PATH** when generating projects with CMake. In addition, append the binary directory of Qt 5 (e.g., **%QT5_DIR%\bin** in windows environment) to the system **PATH**.

- [**Eigen**](http://eigen.tuxfamily.org/) is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Eigen is licensed with Mozilla Public License v.2 (MPL2).
  
  **Eigen 3.2.1** is already included in the repository and it is therefore not needed to install it.

- [**Google Test**](https://code.google.com/p/googletest/) is Google's framework for writing C++ tests on a variety of platforms (Linux, Mac OS X, Windows, Cygwin, Windows CE, and Symbian). Based on the xUnit architecture. Supports automatic test discovery, a rich set of assertions, user-defined assertions, death tests, fatal and non-fatal failures, value- and type-parameterized tests, various options for running the tests, and XML test report generation. Google Test is licensed with New BSD License.
  
  Google Test 1.6.0 is already included in the repository and it is therefore not needed to install it.
