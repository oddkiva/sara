// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Geometry/Tools/Cone.hpp>

namespace DO {
  
  template class Cone<2>;
  template class Cone<3>;
  template class AffineCone<2>;
  template class AffineCone<3>;

} /* namespace DO */