// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! @brief Master header file of the Core module.

#pragma once

// Sara specific defines.
#include <DO/Sara/Defines.hpp>
// Template meta-programming
#include <DO/Sara/Core/Meta.hpp>
// Linear algebra imports and extension from Eigen3
#include <DO/Sara/Core/EigenExtension.hpp>
// N-dimensional array and N-dimensional iterators
#include <DO/Sara/Core/ArrayIterators.hpp>
#include <DO/Sara/Core/MultiArray.hpp>
#include <DO/Sara/Core/Tensor.hpp>
// Sparse N-dimensional array
#include <DO/Sara/Core/SparseMultiArray.hpp>
// Image and color data structures
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Pixel.hpp>
// Tree data structures
#include <DO/Sara/Core/Tree.hpp>
// Timer classes
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Core/TicToc.hpp>
// Miscellaneous
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/Core/StringFormat.hpp>


/*!
  \namespace DO
  @brief The library namespace.

  @defgroup Core Core
  @brief Note that the Core module heavily relies on the
  <a href="http://eigen.tuxfamily.org/">Eigen</a> library (linear algebra).

  the Eigen namespace is directly imported inside the DO namespace for
  convenience. For details, have a look at the file EigenExtension.hpp.

  So more specific information about the Eigen library, refer to:
  http://eigen.tuxfamily.org/dox/
 */
