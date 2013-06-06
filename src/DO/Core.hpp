// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! \brief Master header file of the Core module.

#ifndef DO_CORE_HPP
#define DO_CORE_HPP

// To avoid compilation error with Eigen
#ifdef WIN32
#  define NOMINMAX
#endif

// This is a specific compiling issue with MSVC 2008
#if (_MSC_VER >= 1500 && _MSC_VER < 1600)
# define EIGEN_DONT_ALIGN
# pragma warning ( disable : 4181 ) // "../Core/Locator.hpp(444) : warning C4181: qualifier applied to reference type; ignored"
#endif

//! Activate by default math constants.
#define _USE_MATH_DEFINES

//! Eigen dependencies.
#include <Eigen/Eigen>
// STL dependencies.
#include <algorithm>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stack>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
// System-specific dependencies for "Timer.hpp".
#ifdef _WIN32
# include <windows.h>
#else
# include <sys/time.h>
#endif

// Linear algebra imports and extension from Eigen3
#include "Core/EigenExtension.hpp"
// Template meta-programming
#include "Core/Meta.hpp"
#include "Core/StaticAssert.hpp"
// N-dimensional array and N-dimensional iterators
#include "Core/Locator.hpp"
#include "Core/MultiArray.hpp"
// Sparse N-dimensional array
#include "Core/SparseMultiArray.hpp"
// Image and color data structures
#include "Core/Color.hpp"
#include "Core/Image.hpp"
#include "Core/Subimage.hpp"
// Tree data structures
#include "Core/Tree.hpp"
// Miscellaneous
#include "Core/Stringify.hpp"
#include "Core/Timer.hpp"
#include "Core/StdVectorHelpers.hpp"


/*!
  \namespace DO
  \brief The library namespace.

  \defgroup Core Core
  \brief Note that the Core module heavily relies on the 
  <a href="http://eigen.tuxfamily.org/">Eigen</a> library (linear algebra).

  the Eigen namespace is directly imported inside the DO namespace for 
  convenience. For details, have a look at the file EigenExtension.hpp.

  So more specific information about the Eigen library, refer to:
  http://eigen.tuxfamily.org/dox/
  @{
 */


/*!
  \def srcPath(s)
  \brief Returns the full path of a file when CMake is used.
  
  To make the macro work properly, the file must be be put in the source 
  directory file.
  \param s a C constant string which contains the file name
  \return full path of the file
 
  \def stringSrcPath(s)
  \brief Returns the full path of a file when CMake is used. 
  
  To make the macro work properly, the file must be be put in the source 
  directory file.
  \param s a C++ string which contains the file name
  \return full path of the file
 */
#ifdef SRCDIR
#define SP_STRINGIFY(s)  #s
#define SP_EVAL(s) SP_STRINGIFY(s)
#define srcPath(s) (SP_EVAL(SRCDIR)"/"s)  
#define stringSrcPath(s) (SP_EVAL(SRCDIR)"/"+s)  
#else
#define srcPath(s) (s)
#define stringSrcPath(s) (s)
#endif

//! @}

#endif /* DO_CORE_HPP */