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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>

#include <tinyply-2.2/source/tinyply.h>


namespace DO::Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup MultiviewMisc Miscellaneous I/O
  //! @{

  inline auto save_to_hdf5(H5File& out_h5_file,
                           const TwoViewGeometry& complete_geom,
                           const TensorView_<double, 2>& colors) -> void
  {
    // Get the left and right cameras.
    auto cameras = Tensor_<PinholeCameraDecomposition, 1>{2};
    cameras(0) = complete_geom.C1;
    cameras(1) = complete_geom.C2;

    const MatrixXd X_euclidean = complete_geom.X.colwise().hnormalized();
    SARA_DEBUG << "3D points =\n" << X_euclidean.leftCols(20) << std::endl;

    SARA_DEBUG << "colors =\n" << colors.matrix().topRows(20) << std::endl;
    SARA_DEBUG << "Number of 3D valid points = " << X_euclidean.cols()
               << std::endl;

    SARA_DEBUG << "X.x_coord.min_coeff = " << X_euclidean.col(0).minCoeff()
               << std::endl;
    SARA_DEBUG << "X.x_coord.max_coeff = " << X_euclidean.col(0).maxCoeff()
               << std::endl;
    SARA_DEBUG << "X.y_coord.min_coeff = " << X_euclidean.col(1).minCoeff()
               << std::endl;
    SARA_DEBUG << "X.y_coord.max_coeff = " << X_euclidean.col(1).maxCoeff()
               << std::endl;
    SARA_DEBUG << "X.z_coord.min_coeff = " << X_euclidean.col(2).minCoeff()
               << std::endl;
    SARA_DEBUG << "X.z_coord.max_coeff = " << X_euclidean.col(2).maxCoeff()
               << std::endl;

    SARA_DEBUG << "colors.min_coeff = " << colors.matrix().minCoeff()
               << std::endl;
    SARA_DEBUG << "colors.max_coeff = " << colors.matrix().maxCoeff()
               << std::endl;

    out_h5_file.write_dataset("cameras", cameras, true);
    out_h5_file.write_dataset("points", X_euclidean, true);
    out_h5_file.write_dataset("colors", colors, true);
  }

  inline auto save_to_ply(const TwoViewGeometry& complete_geom,
                          const TensorView_<double, 2>& colors) -> void
  {
    const auto& X = complete_geom.X;
    auto X_data =
        const_cast<double*>(reinterpret_cast<const double*>(X.data()));
    auto X_tensor =
        TensorView_<double, 2>{X_data, {int(complete_geom.X.size()), 3}};

    std::filebuf fb;
    fb.open("/home/david/Desktop/geometry.ply", std::ios::out);
    std::ostream ostr(&fb);
    if (ostr.fail())
      throw std::runtime_error{"Error: failed to create PLY!"};

    tinyply::PlyFile geom_ply_file;
    geom_ply_file.add_properties_to_element(
        "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, X_tensor.size(0),
        reinterpret_cast<std::uint8_t*>(X_tensor.data()),
        tinyply::Type::INVALID, 0);

    auto colors_rgb8 = image_view(colors).convert<Rgb8>();
    geom_ply_file.add_properties_to_element(
        "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8,
        colors.size(0), reinterpret_cast<std::uint8_t*>(colors_rgb8.data()),
        tinyply::Type::INVALID, 0);

    geom_ply_file.write(ostr, false);
  }

  //! @}

} /* namespace DO::Sara */
