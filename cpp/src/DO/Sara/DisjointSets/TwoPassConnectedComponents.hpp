// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/DisjointSets/DisjointSets.hpp>


namespace DO { namespace Sara {

  //! @addtogroup DisjointSets
  //! @{

  DO_SARA_EXPORT
  auto two_pass_connected_components(const ImageView<int, 2>& values)
      -> Image<int, 2>;

  //! @}

} /* namespace Sara */
} /* namespace DO */
