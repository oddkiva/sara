// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/MultiArray/MultiArrayView.hpp>


namespace DO { namespace Shakti {

  //! @brief Resize the image.
  template <typename T, int N, typename Strides>
  void resize(const MultiArrayView<T, N, Strides>& src,
              MultiArrayView<T, N, Strides>& dst)
  {
  }

}}  // namespace DO::Shakti
