// ========================================================================== //
// This file is part of Shakti, a basic set of CUDA accelerated libraries in
// C++ for computer vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/GaussianOctaveComputer.hpp>


namespace DO::Shakti::Cuda {

  GaussianOctaveComputer::GaussianOctaveComputer(int w, int h, int scale_count)
    : host_kernels{scale_count}
    , device_kernels{host_kernels}
    , d_convx{{w, h}}
    , d_octave{make_gaussian_octave<float>(w, h, scale_count)}
  {
    device_kernels.copy_filters_to_device_constant_memory();
  }

  auto GaussianOctaveComputer::operator()(
      MultiArrayView<float, 2, RowMajorStrides>& d_in) -> void
  {
    tic(d_timer);
    device_kernels(d_in, d_convx, d_octave);
    toc(d_timer, "Gaussian Octave");
  }

  auto GaussianOctaveComputer::copy_to_host() -> void
  {
    if (!h_octave.has_value())
      h_octave = Sara::Image<float, 3, PinnedMemoryAllocator>{
          d_octave.width(), d_octave.height(), d_octave.scale_count()};

    // This is an extremely expensive copy.
    tic(d_timer);
    d_octave.array().copy_to(*h_octave);
    toc(d_timer, "Device to Host");
  }

}  // namespace DO::Shakti::Cuda
