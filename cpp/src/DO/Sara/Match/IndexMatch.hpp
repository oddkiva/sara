// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Match.hpp>


namespace DO::Sara {

struct IndexMatch
{
  int i;
  int j;
  float score;
};

template <>
struct CalculateH5Type<IndexMatch>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(IndexMatch)};
    INSERT_MEMBER(h5_comp_type, IndexMatch, i);
    INSERT_MEMBER(h5_comp_type, IndexMatch, j);
    INSERT_MEMBER(h5_comp_type, IndexMatch, score);
    return h5_comp_type;
  }
};

} /* namespace DO::Sara */
