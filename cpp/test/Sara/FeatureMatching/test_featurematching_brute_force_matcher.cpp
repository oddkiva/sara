// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "FeatureMatching/Brute-Force Matcher"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureMatching.hpp>

#if 0
#  ifndef _WIN32
#    ifdef __AVX__
#      include <immintrin.h>
#    else
#      warning AVX is not available. Code will not compile!
#    endif


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestFeatureMatching)

BOOST_AUTO_TEST_CASE(test_dot_product_128)
{
  static const auto dot_product = [](const float* pt1, const float* pt2) {
    // __m256 = 8D vector of `float` numbers.
    __m256 x_dot_y8 = _mm256_setzero_ps();

    // SIFT = 128D vector
    //      = concatenates 16 local HoGs each encoded as an 8D vectors
    //      = 16 x 8D vectors
    //
    // Calculate the 16 dot products and successively accumulate them.
    for (int d = 0; d < 128; d += 8)
    {
      // Load the d-th 8D vectors in v1 in the AVX register
      const __m256 v1 = _mm256_load_ps(pt1 + d);
      // Load the d-th 8D vectors in .
      const __m256 v2 = _mm256_load_ps(pt2 + d);
      // Accumulate the dot product between the two 8D-vectors v1 and v2.
      x_dot_y8 = _mm256_fmadd_ps(v1, v2, x_dot_y8);

      // In short, we just did:
      // x_dot_y8[i] += v1[i] * v2[i], with i in [0, 8[
    }

    // Sum the coefficients of the 8D vectors to get the dot product.
    // We use a trick here...
    // Experiment with godbolt compiler:
    //
    // u =                                  = [  a   b   c   d   e f g h]
    // v = _mm256_permute2f128_ps((v, v, 1) = [  e   f   g   h   a b c d]
    //
    //                                           0   1   2   3
    // w = _mm256_add_ps(u, v)              = [a+e b+f c+g d+h ...]
    //                                               0       1
    // w = _mm256_hadd_ps(u, v)             = [a+e+b+f c+g+d+h ...]
    //                                                       0
    // r = _mm256_hadd_ps(u, v)             = [a+e+b+f+c+g+d+h ...]
    x_dot_y8 = _mm256_add_ps(x_dot_y8,                //
                             _mm256_permute2f128_ps(  //
                                 x_dot_y8, x_dot_y8, 1));
    x_dot_y8 = _mm256_hadd_ps(x_dot_y8, x_dot_y8);

    // This is the dot product.
    const auto x_dot_y = _mm256_cvtss_f32(_mm256_hadd_ps(x_dot_y8, x_dot_y8));
    static_assert(std::is_same_v<decltype(x_dot_y), const float>);
    return x_dot_y;
  };

  auto v1 = Eigen::Matrix<float, 128, 1>{};
  v1.setRandom();

  auto a = float{};
  auto b = float{};
  const auto keys_count = 2000;
  const auto dot_count = keys_count * keys_count;
  a = dot_product(v1.data(), v1.data());
  b = v1.squaredNorm();
  BOOST_CHECK_CLOSE(a, b, std::numeric_limits<float>::epsilon());
}

BOOST_AUTO_TEST_SUITE_END()

#  endif
#else
BOOST_AUTO_TEST_SUITE(TestFeatureMatching)

BOOST_AUTO_TEST_CASE(test_dummy)
{
  // Just to make test pass since we don't have anything at the moment
}

BOOST_AUTO_TEST_SUITE_END()

#endif
