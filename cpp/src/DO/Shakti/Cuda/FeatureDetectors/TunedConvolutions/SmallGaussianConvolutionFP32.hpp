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

#pragma once

#include <DO/Shakti/Cuda/FeatureDetectors/TunedConvolutions/GaussianOctaveKernels.hpp>
#include <DO/Shakti/Cuda/MultiArray/MultiArrayView.hpp>


namespace DO::Shakti::Cuda::Gaussian {

  struct DeviceGaussianFilterBank
  {
    DeviceGaussianFilterBank(const GaussianOctaveKernels<float>& filter_bank)
      : _filter_bank{filter_bank}
    {
    }

    auto copy_filters_to_device_constant_memory() -> void;

    auto peek_filters_in_device_constant_memory() -> void;

    auto operator()(const MultiArrayView<float, 2, RowMajorStrides>& d_in,
                    MultiArrayView<float, 2, RowMajorStrides>& d_convx,
                    MultiArrayView<float, 2, RowMajorStrides>& d_convy,  //
                    int kernel_index) const -> void;

    const GaussianOctaveKernels<float>& _filter_bank;
  };

}  // namespace DO::Shakti::Cuda::Gaussian
