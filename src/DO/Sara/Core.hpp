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

// DO++ specific defines.
#include "Defines.hpp"
// Template meta-programming
#include "Core/Meta.hpp"
#include "Core/StaticAssert.hpp"
// Linear algebra imports and extension from Eigen3
#include "Core/EigenExtension.hpp"
// N-dimensional array and N-dimensional iterators
#include "Core/ArrayIterators.hpp"
#include "Core/MultiArray.hpp"
// Sparse N-dimensional array
#include "Core/SparseMultiArray.hpp"
// Image and color data structures
#include "Core/Pixel.hpp"
#include "Core/Image.hpp"
// Tree data structures
#include "Core/Tree.hpp"
// Timer classes
#include "Core/Timer.hpp"
// Miscellaneous
#include "Core/StdVectorHelpers.hpp"
#include "Core/DebugUtilities.hpp"


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
 */

#endif /* DO_CORE_HPP */