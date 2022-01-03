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

    Output<Buffer<float>> min_dist{"min_dist", 1};
    Output<Buffer<int32_t>> nn{"nearest_neighbors", 1};

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
      auto jj = RDom{0, n2};

      // The L2-distance function.
      auto dist_func = Func{"l2_dist"};
      const auto diff = d1(kk, i) - d2(kk, j);
      dist_func(j, i) = sum(diff * diff);

      Func j_index{"j"};
      j_index(j) = cast<int>(j);

      Func best{"Best"};
      best(i) = argmin(dist_func(jj, i));
      nn(i) = best(i)[0];
      min_dist(i) = best(i)[1];


      // ======================================================================
      // THE SCHEDULE
      // ======================================================================
      auto io = Var{"io"};
      auto ii = Var{"ii"};
      auto jo = Var{"jo"};
      auto ji = Var{"ji"};
      if (get_target().has_gpu_feature())
      {
        // nn.gpu_tile(i, io, ii, 16, TailStrategy::GuardWithIf);

        // min_dist.gpu_tile(i, io, ii, 16, TailStrategy::GuardWithIf);

        best.compute_root();
        // best.update().atomic().parallel(jj);

        // // Calculate all possible distances for a batch of descriptors 1.
        // dist_func.gpu_tile(i, jj, io, jo, ii, ji, 16, 128,
        //                    TailStrategy::GuardWithIf);
        // dist_func.compute_at(min_dist, i);
        // // Unroll the loop in the L2-distance calculation.
        // dist_func.unroll(kk);
      }
      else
      {
        // nn.split(i, io, ii, 8, TailStrategy::GuardWithIf).unroll(ii);
        // nn.unroll(j);

        // min_dist.compute_at(nn, i);
        // min_dist.update().atomic().parallel(jj);
        // // min_dist.unroll(jj);

        // dist_func.compute_at(min_dist, i);
        // // dist_func.split(i, io, ii, 8, TailStrategy::GuardWithIf);
        // // dist_func.split(j, jo, ji, 32,
        // // TailStrategy::GuardWithIf).parallel(jo); dist_func.vectorize(kk, 8);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(BruteForceMatchingL2<float>,
                          shakti_brute_force_nn_l2_32f_gpu)
HALIDE_REGISTER_GENERATOR(BruteForceMatchingL2<float>,
                          shakti_brute_force_nn_l2_32f_cpu)
