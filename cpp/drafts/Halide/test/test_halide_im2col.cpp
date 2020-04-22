// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/Im2Col"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/TensorDebug.hpp>

#include <drafts/Halide/Helpers.hpp>
#include <drafts/Halide/Utilities.hpp>

#include "shakti_im2col_32f.h"

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


using namespace sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_im2col)
{
  constexpr auto N = 3;
  constexpr auto C = 1;
  constexpr auto H = 4;
  constexpr auto W = 3;
  auto x = Tensor_<float, 4>{{N, C, H, W}};

  auto plane = Tensor_<float, 2>{{H, W}};
  plane.matrix() <<
    0,  1,  2,
    3,  4,  5,
    6,  7,  8,
    9, 10, 11;

  x.flat_array() <<            //
      1 * plane.flat_array(),  //
      2 * plane.flat_array(),  //
      3 * plane.flat_array();

  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 1;

  // Apply im2col on each data of the batch.
  auto phi_x = Tensor_<float, 2>{{N * H * W, kH * kW * kC}};
  auto phi_x_3d = phi_x.reshape(Vector3i{N, H * W, kH * kW * kC});

  auto x_buffer = halide::as_runtime_buffer(x);
  auto phi_x_buffer = halide::as_runtime_buffer(phi_x_3d);

  x_buffer.set_host_dirty();
  shakti_im2col_32f(x_buffer, -1, -1, 0, kW, kH, 1, phi_x_buffer);
  phi_x_buffer.copy_to_host();

  // Check the reshaped im2col
  auto sizes_5d = Matrix<int, 5, 1>{};
  sizes_5d << N, H, W, kH, kW;
  auto phi_x_as_5d = phi_x.reshape(sizes_5d);

  MatrixXf true_neighborhood(kH, kW);
  true_neighborhood <<
    0, 0, 0,
    1, 2, 0,
    4, 5, 0;
  //                      n  y  x
  BOOST_CHECK(phi_x_as_5d[0][0][2].matrix() == true_neighborhood);
  // cout << phi_x_as_5d[0][0][2].matrix() << endl << endl;

  true_neighborhood <<
    0, 0, 0,
    0, 0, 2,
    0, 6, 8;
  BOOST_CHECK(phi_x_as_5d[1][0][0].matrix() == true_neighborhood);
  // cout << phi_x_as_5d[1][0][0].matrix() << endl << endl;

  true_neighborhood <<
    2 * 3, 2 * 4, 2 * 5,
    2 * 6, 2 * 7, 2 * 8,
    2 * 9, 2 *10, 2 *11;
  BOOST_CHECK(phi_x_as_5d[1][2][1].matrix() == true_neighborhood);
  // cout << phi_x_as_5d[1][2][1].matrix() << endl << endl;

  true_neighborhood <<
    3 * 3, 3 * 4, 3 * 5,
    3 * 6, 3 * 7, 3 * 8,
    3 * 9, 3 *10, 3 *11;
  BOOST_CHECK(phi_x_as_5d[2][2][1].matrix() == true_neighborhood);
  // cout << phi_x_as_5d[2][2][1].matrix() << endl << endl;
}
