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

// Because of Eigen/Core compiling error
#ifdef WIN32
#	define NOMINMAX
#endif

// This is a specific compiling issue with MSVC 2008
#if (_MSC_VER >= 1500 && _MSC_VER < 1600)
# define EIGEN_DONT_ALIGN
# pragma warning ( disable : 4181 ) // "../Core/Locator.hpp(444) : warning C4181: qualifier applied to reference type; ignored"
#endif

//! Activate by default math constants.
#define _USE_MATH_DEFINES

#include <Eigen/Eigen>

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

#include "Core/EigenExtension.hpp"
#include "Core/Meta.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Stringify.hpp"
#include "Core/Timer.hpp"
#include "Core/Color.hpp"
#include "Core/Locator.hpp"
#include "Core/MultiArray.hpp"
#include "Core/SparseMultiArray.hpp"
#include "Core/Image.hpp"
#include "Core/Tree.hpp"

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

//! Opening macro whose sole purpose is to increase the namespace visibility.
#define BEGIN_NAMESPACE_DO namespace DO {
//! Closing macro whose sole purpose is to increase the namespace visibility.
#define END_NAMESPACE_DO } /* namespace DO */

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