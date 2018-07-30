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

#define BOOST_TEST_MODULE "ImageProcessing/GEMM-based Convolution"

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <sstream>


using namespace std;
using namespace DO::Sara;


template <typename T>
void print_3d_array(const TensorView_<T, 3>& x)
{
  const auto max = x.flat_array().abs().maxCoeff();
  std::stringstream ss;
  ss << max;
  const auto pad_size = ss.str().size();


  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << std::setw(pad_size) << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}


BOOST_AUTO_TEST_CASE(test_im2col)
{
  constexpr auto N = 3;
  constexpr auto H = 4;
  constexpr auto W = 3;
  auto x = Tensor_<float, 3>{{N, H, W}};

  auto plane = Tensor_<float, 2>{{H, 3}};
  plane.matrix() <<
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9,10,11;

  x.flat_array() <<            //
      1 * plane.flat_array(),  //
      2 * plane.flat_array(),  //
      3 * plane.flat_array();

  constexpr auto kH = 3;
  constexpr auto kW = 3;

  auto phi_x = Tensor_<float, 2>{{N * H * W, kH * kW}};
  auto phi_x_as_3d = phi_x.reshape(Vector3i{N, H* W, kH * kW});

  // Apply im2col on each plane.
  phi_x_as_3d[0] = im2col(x[0], {kH, kW});
  phi_x_as_3d[1] = im2col(x[1], {kH, kW});
  phi_x_as_3d[2] = im2col(x[2], {kH, kW});

  // Apply im2col on the whole batch.
  auto phi_x_2 = im2col(x, {1, kH, kW});

  BOOST_CHECK(phi_x.sizes() == phi_x_2.sizes());
  BOOST_CHECK(phi_x.matrix() == phi_x_2.matrix());

  auto sizes_5d = Matrix<int, 5, 1>{};
  sizes_5d << N, H, W, kH, kW;
  auto phi_x_as_5d = phi_x.reshape(sizes_5d);

  cout << phi_x_as_5d[0][0][2].matrix() << endl << endl;

  cout << phi_x_as_5d[1][0][0].matrix() << endl << endl;
  cout << phi_x_as_5d[1][2][1].matrix() << endl << endl;
  cout << phi_x_as_5d[1][2][1].matrix() << endl << endl;
}

BOOST_AUTO_TEST_CASE(test_im2col_strided_on_nhwc_tensor)
{
  constexpr auto N = 1;
  constexpr auto H = 4;
  constexpr auto W = 3;
  constexpr auto C = 3;
  auto x = Tensor_<float, 4>{{N, H, W, C}};

  x[0].flat_array() <<
    0,0,0,  1, 1, 1,  2, 2, 2,
    3,3,3,  4, 4, 4,  5, 5, 5,
    6,6,6,  7, 7, 7,  8, 8, 8,
    9,9,9, 10,10,10, 11,11,11;

  for (int i = 1; i < N; ++i)
    x[i].flat_array() = x[0].flat_array(); //(i + 1) * x[i - 1].flat_array();

  print_3d_array(x[0]);

  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;

  // Apply im2col on the whole batch.
  //                             kernel_sizes     strides
  //                              N  H   W   C
  // Each row (y, x) describes a patch centered at (y, x) with sizes (3, 3, 3).
  //
  // 3D patch centered at (n=0, y=1, x=1).
  // ROW 0
  // [[(y-1, x-1, c-1), (y-1, x-1, c), (y-1, x-1, c+1)],
  //  [(y-1, x  , c-1), (y-1, x  , c), (y-1, x  , c+1)],
  //  [(y-1, x+1, c-1), (y-1, x+1, c), (y-1, x+1, c+1)]],
  //
  // ROW 1
  // [[(y  , x-1, c-1), (y  , x-1, c), (y  , x-1, c+1)],
  //  [(y  , x  , c-1), (y  , x  , c), (y  , x  , c+1)],
  //  [(y  , x+1, c-1), (y  , x+1, c), (y  , x+1, c+1)]],
  //
  // ROW 2
  // [[(y  , x-1, c-1), (y  , x-1, c), (y  , x-1, c+1)],
  //  [(y  , x  , c-1), (y  , x  , c), (y  , x  , c+1)],
  //  [(y  , x+1, c-1), (y  , x+1, c), (y  , x+1, c+1)]],
  //
  // Adding the non-zero shift (0, 0, 1) in im2col will do
  // ROW 0
  // [[(y-1, x-1, c), (y-1, x-1, c+1), (y-1, x-1, c+2)],
  //  [(y-1, x  , c), (y-1, x  , c+1), (y-1, x  , c+2)],
  //  [(y-1, x+1, c), (y-1, x+1, c+1), (y-1, x+1, c+2)]],
  //
  // ROW 1
  // [[(y  , x-1, c), (y  , x-1, c+1), (y  , x-1, c+2)],
  //  [(y  , x  , c), (y  , x  , c+1), (y  , x  , c+2)],
  //  [(y  , x+1, c), (y  , x+1, c+1), (y  , x+1, c+2)]],
  //
  // ROW 2
  // [[(y  , x-1, c), (y  , x-1, c+1), (y  , x-1, c+2)],
  //  [(y  , x  , c), (y  , x  , c+1), (y  , x  , c+2)],
  //  [(y  , x+1, c), (y  , x+1, c+1), (y  , x+1, c+2)]],
  auto phi_x = im2col_strided(x, {1, kH, kW, kC}, {1, 1, 1, 3}, {0, 0, 0, 1});

  BOOST_CHECK(phi_x.sizes() == Vector2i(N * H * W, kH * kW * kC));
  //BOOST_CHECK(phi_x.matrix() == phi_x_2.matrix());

  auto sizes_6d = Matrix<int, 6, 1>{};
  sizes_6d << N, H, W, kH, kW, kC;
  auto phi_x_as_6d = phi_x.reshape(sizes_6d);

  print_3d_array(phi_x_as_6d[0][1][1]);
  print_3d_array(phi_x_as_6d[0][0][0]);
}

BOOST_AUTO_TEST_CASE(test_convolve)
{
  constexpr auto N = 10;
  constexpr auto H = 4;
  constexpr auto W = 6;
  constexpr auto C = 3;
  auto x = Tensor_<float, 4>{{N, C, H, W}};
  x.flat_array().fill(1.f);

  constexpr auto kN = 5;
  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;
  auto k = Tensor_<float, 4>{{kN, kC, kH, kW}};

  k.flat_array() <<
    // R
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    // G
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    // B
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,

    // R
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
    // G
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,
    // B
    2, 2, 2,
    2, 2, 2,
    2, 2, 2,

    // R
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
    // G
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,
    // B
    3, 3, 3,
    3, 3, 3,
    3, 3, 3,

    // R
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,
    // G
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,
    // B
    4, 4, 4,
    4, 4, 4,
    4, 4, 4,

    // R
    5, 5, 5,
    5, 5, 5,
    5, 5, 5,
    // G
    5, 5, 5,
    5, 5, 5,
    5, 5, 5,
    // B
    5, 5, 5,
    5, 5, 5,
    5, 5, 5;

  auto y = x;
  gemm_convolve(y, x, k);
}

BOOST_AUTO_TEST_CASE(test_convolve_strided_on_nhwc_tensor)
{
  constexpr auto N = 1;
  constexpr auto H = 4;
  constexpr auto W = 3;
  constexpr auto C = 3;
  auto x = Tensor_<float, 4>{{N, H, W, C}};

  x[0].flat_array() <<
    0,0,0,   1, 1, 1,   2, 2, 2,
    3,3,3,   4, 4, 4,   5, 5, 5,
    6,6,6,   7, 7, 7,   8, 8, 8,
    9,9,9,  10,10,10,  11,11,11;

  for (int i = 1; i < N; ++i)
    x[i].flat_array() = x[0].flat_array(); //(i + 1) * x[i - 1].flat_array();

  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;

  auto phi_x = im2col_strided(x, {N, kH, kW, kC}, {1, 1, 1, kC}, {0, 0, 0, 1});
  cout << "phi = " << phi_x.matrix().rows() << " " << phi_x.matrix().cols()  << endl;
  cout << phi_x.matrix() << endl;

  //                   kH x kW x kI  kO
  Tensor_<float, 2> k{{ 3 *  3 *  3,  3}};

  // Average on the R channel.
  k.matrix().col(0) <<
    1, 0, 0,  1, 0, 0,  1, 0, 0,
    1, 0, 0,  1, 0, 0,  1, 0, 0,
    1, 0, 0,  1, 0, 0,  1, 0, 0;

  // Average on the G channel.
  k.matrix().col(1) <<
    0, 1, 0,  0, 1, 0,  0, 1, 0,
    0, 1, 0,  0, 1, 0,  0, 1, 0,
    0, 1, 0,  0, 1, 0,  0, 1, 0;

  // Average on the B channel.
  k.matrix().col(2) <<
    0, 0, 1,  0, 0, 1,  0, 0, 1,
    0, 0, 1,  0, 0, 1,  0, 0, 1,
    0, 0, 1,  0, 0, 1,  0, 0, 1;
  k.flat_array() /= 9;

  cout << "k = " << k.matrix().rows() << " " << k.matrix().cols()  << endl;
  cout << k.matrix()  << endl;

  auto y = Tensor_<float, 4>{{N, C, H, W}};
  y.flat_array() = (phi_x.matrix() * k.matrix()).array();

  auto yt = y.transpose({0, 2, 3, 1});

  print_3d_array(y[0]);
  print_3d_array(yt[0]);
}

BOOST_AUTO_TEST_CASE(test_convolve_strided_on_nchw_tensor)
{
  constexpr auto N = 1;
  constexpr auto H = 4;
  constexpr auto W = 3;
  constexpr auto C = 3;
  auto x = Tensor_<float, 4>{{N, C, H, W}};

  x[0].flat_array() <<
    0,   1,   2,
    3,   4,   5,
    6,   7,   8,
    9,  10,  11;

  for (int i = 1; i < C; ++i)
    x[0][i].flat_array() = x[0][0].flat_array();

  cout << x[0][0].matrix() << endl;


  constexpr auto kH = 3;
  constexpr auto kW = 3;
  constexpr auto kC = 3;


  auto phi_x = im2col_strided(x, {N, kC, kH, kW}, {1, kC, 1, 1}, {0, 1, 0, 0});
  // [N * C/kC * H/kH * W/kW, kC * kH * kW]
  cout << "phi = " << phi_x.matrix().rows() << " " << phi_x.matrix().cols()  << endl;
  cout << phi_x.matrix() << endl;

  //                   kC x kH x kW  kO
  Tensor_<float, 2> k{{ 3 *  3 *  3,  3}};
  k.matrix().col(0) << VectorXf::Ones(9) / 9, VectorXf::Zero(9)    , VectorXf::Zero(9);
  k.matrix().col(1) << VectorXf::Zero(9)    , VectorXf::Ones(9) / 9, VectorXf::Zero(9);
  k.matrix().col(2) << VectorXf::Zero(9)    , VectorXf::Zero(9)    , VectorXf::Ones(9) / 9;

  cout << "k = " << k.matrix().rows() << " " << k.matrix().cols()  << endl;
  cout << k.matrix()  << endl;

  auto y = Tensor_<float, 4>{{N, C, H, W}};
  y.flat_array() = (phi_x.matrix() * k.matrix()).array();
  y = y.transpose(Vector4i{0, 2, 3, 1});

  print_3d_array(y[0]);
}
