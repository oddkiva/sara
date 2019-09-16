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

#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>


namespace DO::Sara {

  std::ostream& operator<<(std::ostream& os, const Homography& H)
  {
    return os << H.matrix() << std::endl;
  }

} /* namespace DO::Sara */
