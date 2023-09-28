// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/FeatureDetectors/SIFT/Octave.hpp>


namespace DO::Shakti::Cuda::v2 {

  template <typename T>
  class Pyramid
  {
  public:
    using octave_type = Octave<T>;

    inline Pyramid() noexcept = default;

    inline auto operator[](int i) -> auto&
    {
      return _octaves[i];
    }

    inline auto operator[](int i) const -> const auto&
    {
      return _octaves[i];
    }

  private:
    std::vector<octave_type> _octaves;
  };

}  // namespace DO::Shakti::Cuda::v2
