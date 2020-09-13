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

#include <drafts/Halide/Components/StreamCompaction.hpp>


namespace {

  using namespace Halide;
  namespace halide = DO::Shakti::HalideBackend;

  class StreamCompaction : public Generator<StreamCompaction>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 256};

    Input<Buffer<std::int8_t>> input{"input", 4};
    Output<Buffer<std::int32_t>[4]> output{"output", 1};

    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};

    void generate()
    {
      const auto w = input.dim(0).extent();
      const auto h = input.dim(1).extent();
      const auto c = input.dim(2).extent();
      const auto n = input.dim(3).extent();
      const auto size = w * h * c * n;

      auto in_flattened = Halide::Func{"input_flattened"};
      in_flattened(x) =
          input(x % w, (x / w) % h, (x / (w * h)) % c, x / (w * h * c));

      // Express the prefix sum by a recurrence relation.
      auto prefix_sum = Halide::Func{"prefix_sum"};
      prefix_sum(x) = cast<int32_t>(0);

      RDom range(1, size - 1);
      prefix_sum(range) = select(in_flattened(range - 1) != 0,    //
                                 prefix_sum(range - 1) + 1,  //
                                 prefix_sum(range - 1));
      prefix_sum.compute_root();

      // Compacted indices.
      RDom in_range(0, size);
      auto compacted_indices = Halide::Func{"compacted_indices"};
      compacted_indices(x) = -1;
      compacted_indices(clamp(prefix_sum(in_range), 0, size - 1)) =
          select(in_flattened(in_range) != 0, in_range, -1);
      compacted_indices.compute_root();

      auto& x_indices = output[0];
      auto& y_indices = output[1];
      auto& c_indices = output[2];
      auto& n_indices = output[3];

      const auto flat_index = compacted_indices(x);
      x_indices(x) = select(flat_index != -1, flat_index % w, -1);
      y_indices(x) = select(flat_index != -1, (flat_index / w) % h, -1);
      c_indices(x) = select(flat_index != -1, (flat_index / (w * h)) % c, -1);
      n_indices(x) = select(flat_index != -1, flat_index / (w * h * c), -1);


      // // GPU schedule.
      // if (get_target().has_gpu_feature())
      // {
      //   // x_indices.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
      //   // y_indices.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
      //   // c_indices.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
      //   // n_indices.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
      // }

      // // Hexagon schedule.
      // else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      // {
      //   const auto vector_size =
      //       get_target().has_feature(Target::HVX_128) ? 128 : 64;

      //   output.hexagon()
      //       .split(x, xo, xi, 128)
      //       .parallel(xo)
      //       .vectorize(xi, vector_size, TailStrategy::GuardWithIf);
      // }

      // // CPU schedule.
      // else
      // {
      //   output.split(x, xo, xi, 8)
      //       .parallel(xo)
      //       .vectorize(xi, 8, TailStrategy::GuardWithIf);
      // }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(StreamCompaction, shakti_stream_compaction)
