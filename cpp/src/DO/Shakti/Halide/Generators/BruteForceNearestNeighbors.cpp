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

#include <DO/Shakti/Halide/MyHalide.hpp>


namespace {

  using namespace Halide;

  template <typename T>
  class BruteForceMatchingL2 : public Generator<BruteForceMatchingL2<T>>
  {
  public:
    using Base = Generator<BruteForceMatchingL2<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

    Input<Buffer<T>> d1{"descriptors_1", 2};
    Input<Buffer<T>> d2{"descriptors_2", 2};

    Output<Buffer<int32_t>> nn1{"nn1", 1};
    Output<Buffer<int32_t>> nn2{"nn2", 1};
    Output<Buffer<float>> dist1{"min_dist1", 1};
    Output<Buffer<float>> dist2{"min_dist2", 1};

    void generate()
    {
      const auto n1 = d1.dim(1).extent();
      const auto n2 = d2.dim(1).extent();
      const auto k = d1.dim(0).extent();

      // ======================================================================
      // THE ALGORITHM
      // ======================================================================
      auto i = Var{"i"};
      auto j = Var{"j"};
      auto kk = RDom{0, k};
      auto jj = RDom{2, n2 - 2};

      // The L2-distance function.
      auto dist_func = Func{"l2_distance"};
      const auto diff = d1(kk, i) - d2(kk, j);
      dist_func(j, i) = sum(diff * diff);

      Func nn2_func{"two_nearest_neighbors"};

      Expr init_nn_1 = select(dist_func(0, i) < dist_func(1, i), 0, 1);
      Expr init_nn_2 = select(dist_func(0, i) < dist_func(1, i), 1, 0);

      Expr init_dist_1 = select(dist_func(0, i) < dist_func(1, i),
                                dist_func(0, i), dist_func(1, i));
      Expr init_dist_2 = select(dist_func(0, i) < dist_func(1, i),
                                dist_func(1, i), dist_func(0, i));

      nn2_func(i) = Tuple{init_nn_1, init_nn_2, init_dist_1, init_dist_2};

      Expr old_nn1 = nn2_func(i)[0];
      Expr old_nn2 = nn2_func(i)[1];
      Expr old_min1 = nn2_func(i)[2];
      Expr old_min2 = nn2_func(i)[3];

      Expr new_nn1 = select(dist_func(jj, i) < old_min1, jj, old_nn1);
      Expr new_nn2 = select(dist_func(jj, i) < old_min1, old_nn1, old_nn2);

      Expr new_min1 = select(dist_func(jj, i) < old_min1, dist_func(jj, i), old_min1);
      Expr new_min2 = select(dist_func(jj, i) < old_min1, old_min1, old_min2);

      nn2_func(i) = {new_nn1, new_nn2, new_min1, new_min2};

      nn1(i) = nn2_func(i)[0];
      nn2(i) = nn2_func(i)[1];
      dist1(i) = nn2_func(i)[2];
      dist2(i) = nn2_func(i)[3];


      // ======================================================================
      // THE SCHEDULE
      //
      // HOWTO? That is the question...
      // ======================================================================
      auto io = Var{"io"};
      auto ii = Var{"ii"};
      auto ko = Var{"jo"};
      auto ki = Var{"ji"};
      if (get_target().has_gpu_feature())
      {
        nn2_func.gpu_tile(i, io, ii, 16, TailStrategy::GuardWithIf);
      }
      else
      {
        nn2_func.split(i, io, ii, 8, TailStrategy::GuardWithIf).parallel(io).unroll(ii);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(BruteForceMatchingL2<float>,
                          shakti_brute_force_nn_l2_32f_gpu)
HALIDE_REGISTER_GENERATOR(BruteForceMatchingL2<float>,
                          shakti_brute_force_nn_l2_32f_cpu)
