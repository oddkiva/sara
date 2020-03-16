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

#include <cuda_runtime_api.h>
#include <nppdefs.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>


namespace DO { namespace Shakti {

  //! @brief Resize the image.
  template <typename Strides>
  void resize(const MultiArrayView<std::uint8_t, 3, Strides>& src,
              MultiArrayView<std::uint8_t, 3, Strides>& dst,
              NppiInterpolationMode interpolation_mode = NPPI_INTER_CUBIC)
  {
    // Resize the image to the appropriate size
    const auto status =
        nppiResize_8u_C3R(static_cast<const Npp8u*>(src.data()),  // src pointer
                          src.width() * 3,                        // src pitch
                          {src.width(), src.height()},            // src size
                          {0, 0, src.width(), src.height()},      // src roi

                          static_cast<Npp8u*>(dst.data()),    // dst pointer
                          dst.width() * 3,                    // dst pitch
                          {dst.width(), dst.height()},        // dst size
                          {0, 0, dst.width(), dst.height()},  // dst roi

                          interpolation_mode);  // interpolation type.

    if (status != NPP_SUCCESS)
      throw std::runtime_error{"Error resizing image!"};
  }

  //! @brief Resize the image.
  template <typename Strides>
  void resize(const MultiArrayView<float, 3, Strides>& src,
              MultiArrayView<float, 3, Strides>& dst,
              NppiInterpolationMode interpolation_mode = NPPI_INTER_CUBIC)
  {
    // Resize the image to the appropriate size
    const auto status = nppiResize_32f_C3R(
        static_cast<const Npp32f*>(src.data()),  // src pointer
        src.width() * 3,                         // src pitch
        {src.width(), src.height()},             // src size
        {0, 0, src.width(), src.height()},       // src roi

        static_cast<Npp32f*>(dst.data()),   // dst pointer
        dst.width() * 3,                    // dst pitch
        {dst.width(), dst.height()},        // dst size
        {0, 0, dst.width(), dst.height()},  // dst roi

        interpolation_mode);  // interpolation type.

    if (status != NPP_SUCCESS)
      throw std::runtime_error{"Error resizing image!"};
  }

  //! @brief Resize the image.
  template <typename Strides>
  void resize(const MultiArrayView<double, 3, Strides>& src,
              MultiArrayView<double, 3, Strides>& dst,
              NppiInterpolationMode interpolation_mode = NPPI_INTER_CUBIC)
  {
    // Resize the image to the appropriate size
    const auto status = nppiResize_64f_C3R(
        static_cast<const Npp32f*>(src.data()),  // src pointer
        src.width() * 3,                         // src pitch
        {src.width(), src.height()},             // src size
        {0, 0, src.width(), src.height()},       // src roi

        static_cast<Npp32f*>(dst.data()),   // dst pointer
        dst.width() * 3,                    // dst pitch
        {dst.width(), dst.height()},        // dst size
        {0, 0, dst.width(), dst.height()},  // dst roi

        interpolation_mode);  // interpolation type.

    if (status != NPP_SUCCESS)
      throw std::runtime_error{"Error resizing image!"};
  }

}}  // namespace DO::Shakti
