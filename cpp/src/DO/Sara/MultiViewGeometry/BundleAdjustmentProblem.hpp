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

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Logging/Logger.hpp>

#include <vector>


namespace DO::Sara {

  /*!
   *  @ingroup MultiViewGeometry
   *  @defgroup MultiviewBA Bundle Adjustment
   *  @{
   */

  //! @brief Bundle adjustment class.
  struct BundleAdjustmentData
  {
    //! @brief An observation is a 2D image point.
    Tensor_<double, 2> observations;
    //! @brief the corresponding 3D point index for the observation 'o'
    std::vector<int> point_indices;
    //! @brief the corresponding 3D camera index for the observation 'o'
    std::vector<int> camera_indices;


    //! @brief The parameter is the set of camera parameters and the set of 3D
    //! point coordinates.
    std::vector<double> parameters;
    //! @{
    //! @brief The number of parameters in details.
    int num_cameras;
    int num_intrinsics;
    int num_extrinsics;
    int num_points;
    //! @}
    //! @brief Convenient parameter data views.
    TensorView_<double, 2> intrinsics;
    TensorView_<double, 2> extrinsics;
    TensorView_<double, 2> point_coords;

    auto resize(const int num_image_points, const int num_scene_points,
                const int num_views, const int num_intrinsic_params,
                const int num_extrinsic_params)
    {
      auto& logger = Logger::get();

      SARA_LOGI(logger, "Allocating memory for observation data...");
      observations = Tensor_<double, 2>{{num_image_points, 2}};
      point_indices = std::vector<int>(num_image_points);
      camera_indices = std::vector<int>(num_image_points);

      SARA_LOGI(logger, "Allocating memory for parameter data...");
      // Store the number of parameters.
      num_cameras = num_views;
      num_intrinsics = num_intrinsic_params;
      num_extrinsics = num_extrinsic_params;
      num_points = num_scene_points;
      // Allocate.
      const auto num_parameters =
          num_cameras * (num_intrinsics + num_extrinsics) + num_points * 3;
      parameters = std::vector<double>(num_parameters);

      // Update the memory views.
      auto intrinsics_new = TensorView_<double, 2>{
          parameters.data(),             //
          {num_cameras, num_extrinsics}  //
      };
      auto extrinsics_new = TensorView_<double, 2>{
          parameters.data() + num_cameras * num_intrinsics,  //
          {num_cameras, num_extrinsics}                      //
      };
      auto point_coords_new = TensorView_<double, 2>{
          parameters.data() + num_cameras * (num_extrinsics + num_intrinsics),
          {num_points, 3}};

      intrinsics.swap(intrinsics_new);
      extrinsics.swap(extrinsics_new);
      point_coords.swap(point_coords_new);
    }
  };

  //! @}

} /* namespace DO::Sara */
