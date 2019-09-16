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

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>


namespace DO::Sara {

DO_SARA_EXPORT
auto estimate_two_view_geometry(const TensorView_<int, 2>& M,
                                const TensorView_<double, 2>& un1,
                                const TensorView_<double, 2>& un2,
                                const EssentialMatrix& E,
                                const TensorView_<bool, 1>& inliers,
                                const TensorView_<int, 1>& sample_best)
    -> TwoViewGeometry;

DO_SARA_EXPORT
auto keep_cheiral_inliers_only(TwoViewGeometry& geometry,
                               const TensorView_<bool, 1>& inliers) -> void;

DO_SARA_EXPORT
auto extract_colors(const Image<Rgb8>& image1,             //
                    const Image<Rgb8>& image2,             //
                    const TwoViewGeometry& complete_geom)  //
    -> Tensor_<double, 2>;

} /* namespace DO::Sara */
