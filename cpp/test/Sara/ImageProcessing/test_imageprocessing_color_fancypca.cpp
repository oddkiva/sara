// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Color Fancy PCA"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/ColorFancyPCA.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_color_fancy_pca)
{
  auto image = Image<Rgb32f>{2, 2};
  image.matrix().fill(Rgb32f::Zero());

  auto alpha = Vector3f::Ones().eval();

  auto fancy_pca = ColorFancyPCA{Matrix3f::Identity(), Vector3f::Ones()};
  fancy_pca(image, alpha);

  for (int i = 0; i < 3; ++i)
    BOOST_CHECK_EQUAL(to_cwh_tensor(image)[0].matrix(), Matrix2f::Ones());
}
