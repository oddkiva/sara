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

#pragma once

#include <vector>


//! @{
//! Helper functions for std::vector.
template <typename T>
inline void append(std::vector<T>& a, const std::vector<T>& b)
{
  a.insert(a.end(), b.begin(), b.end());
}

template <typename T>
inline void shrink_to_fit(std::vector<T>& v)
{
  std::vector<T>(v).swap(v);
}
//! @}
