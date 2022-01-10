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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Timer.hpp>

#include <DO/Sara/FeatureMatching/BruteForceMatcher.hpp>

#ifdef __AVX__
#  include <immintrin.h>
#else
#  warning AVX is not available. Code will not compile!
#endif


using namespace std;


namespace DO::Sara {

  BruteForceMatcher::BruteForceMatcher(
      const KeypointList<OERegion, float>& keys1,
      const KeypointList<OERegion, float>& keys2,  //
      float sift_ratio_thres)
    : _keys1(keys1)
    , _keys2(keys2)
    , _squared_ratio_thres(sift_ratio_thres * sift_ratio_thres)
  {
    if (!size_consistency_predicate(_keys1) ||
        !size_consistency_predicate(_keys2))
      throw std::runtime_error{"The number of interest points and descriptors "
                               "are not mutually equal!"};
  }

  //! Compute candidate matches using the Euclidean distance.
  auto BruteForceMatcher::compute_matches() -> vector<Match>
  {
    auto t = Timer{};

    auto matches = vector<Match>{};
    matches.reserve(1e5);

    auto nn = Tensor_<int, 2>{{size(_keys1), 2}};
    auto square_distances = Tensor_<float, 2>{{size(_keys1), 2}};

    // Brute-Force SIFT matcher.
    static constexpr auto BSIZ = 256;  // Block size (batch size).
    static constexpr auto NDIM = 128;  // Dimension of the SIFT descriptor.

    // Focus on matchng keys 1 to keys 2 for now.
    const auto npts1 = size(_keys1);
    const auto npts2 = size(_keys2);

    const auto& dmat1 = descriptors(_keys1);
    const auto& dmat2 = descriptors(_keys2);

    // Initialize the nearest neighbors.
#pragma omp parallel for
    for (int i1 = 0; i1 < npts1; ++i1)
    {
      const auto d1 = (dmat1[i1].vector() - dmat2[0].vector()).squaredNorm();
      const auto d2 = (dmat1[i1].vector() - dmat2[1].vector()).squaredNorm();

      if (d1 <= d2)
      {
        nn(i1, 0) = 0;
        nn(i1, 1) = 1;
        square_distances(i1, 0) = d1;
        square_distances(i1, 1) = d2;
      }
      else
      {
        nn(i1, 0) = 1;
        nn(i1, 1) = 0;
        square_distances(i1, 0) = d2;
        square_distances(i1, 1) = d1;
      }
    }

    // Marten does not make any tail strategies that takes care of cases where
    // the size of keypoints are not multiples of 256.
    //
    // So we need to take care of that.

    const auto& pts1 = dmat1.data();
    const auto& pts2 = dmat2.data();

    static const auto dot_product = [](const float* pt1, const float* pt2) {
      // __m256 = 8D vector of `float` numbers.
      __m256 x_dot_y8 = _mm256_setzero_ps();

      // SIFT = 128D vector
      //      = concatenates 16 local HoGs each encoded as an 8D vectors
      //      = 16 x 8D vectors
      //
      // Calculate the 16 dot products and successively accumulate them.
      for (int d = 0; d < NDIM; d += 8)
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

#pragma omp parallel for
    for (int b1 = 0; b1 < npts1; b1 += BSIZ)
    {
      // For each block of 256 SIFT descriptors in the first set of keypoints
      // b1 + 0  ...    b1 + 256

      for (int b2 = 2; b2 < npts1; b2 += BSIZ)
      {
        // For each block of 256 SIFT descriptors in the second set of keypoints
        // b2 + 0  ...    b2 + 256

        for (int p1 = b1; p1 < b1 + BSIZ; ++p1)
        {
          if (p1 >= npts1)
            continue;

          const auto v1 = &pts1[p1 * NDIM];
          // For each vector pt1 in the block [b1, b1 + 256[
          //   calculate its dot product with pt2 in the block [b2, b2 + 256[

          for (int p2 = b2; p2 < b2 + BSIZ; ++p2)
          {
            if (p2 >= npts2)
              continue;

            const auto v2 = &pts2[p2 * NDIM];

            // The dot product
            const auto v1_dot_v2 = dot_product(v1, v2);
            static_assert(std::is_same_v<decltype(v1_dot_v2), const float>);

            // By design, SIFT descriptors are actually 128D unit vectors.
            // The L2-distance between two normalized vectors x and y:
            // |x - y|^2 = |x|^2 + |y|^2 - 2 <x, y>
            //
            // Because |x| = |y| = 1, the L2-distance is:
            // |x - y|^2 =  2 * (1 - <x, y>)
            const auto square_dist = 2 * (1 - v1_dot_v2);

#ifdef USE_GENERIC_ALGO
            static constexpr auto K = 2;
            for (int nn = 0; nn < K; ++nn)
            {
              if (square_dist < square_distances(p1, nn))
              {
                for (auto j = K - 1; j > nn; ++j)
                {
                  squared_distances(p1, j) = squared_distances(p1, j - 1);
                  nn(p1, j) = nn(p1, j - 1);
                }
                squared_distances(p1, nn) = squared_dist;
                nn(p1, nn) = p2;

                break;
              }
            }
#else
            if (square_dist < square_distances(p1, 0))
            {
              // Update the 2-NN.
              square_distances(p1, 1) = square_distances(p1, 0);
              nn(p1, 1) = nn(p1, 0);

              // Update the 1-NN
              square_distances(p1, 0) = square_dist;
              nn(p1, 0) = p2;
            }
            else if (square_dist < square_distances(p1, 1))
            {
              // Update the 2-NN.
              square_distances(p1, 1) = square_dist;
              nn(p1, 1) = p2;
            }
#endif
          }
        }
      }
    }

    // Remove redundant matches in each consecutive group of identical
    // matches. We keep the one with the best Lowe score.
    matches.resize(unique(matches.begin(), matches.end()) - matches.begin());

    // Reorder the matches again.
    sort(matches.begin(), matches.end(), [&](const Match& m1, const Match& m2) {
      return m1.score() < m2.score();
    });

    SARA_DEBUG << "Computed " << matches.size() << " matches in " << t.elapsed()
               << " seconds." << endl;

    return matches;
  }

}  // namespace DO::Sara
