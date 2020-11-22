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
    Output<Buffer<std::int32_t>[4]> xysn{"xysn", 1};
    Output<Buffer<std::int8_t>> value{"value", 1};

    Var x{"x"};
    Var xo{"xo"};
    Var xi{"xi"};

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

      auto x_index_fn = Halide::Func{"x_index"};
      auto y_index_fn = Halide::Func{"y_index"};
      auto s_index_fn = Halide::Func{"s_index"};
      auto n_index_fn = Halide::Func{"n_index"};
      auto value_fn = Halide::Func{"value_fn"};

      const auto flat_index = compacted_indices(x);
      x_index_fn(x) = select(flat_index != -1, flat_index % w, -1);
      y_index_fn(x) = select(flat_index != -1, (flat_index / w) % h, -1);
      s_index_fn(x) = select(flat_index != -1, (flat_index / (w * h)) % c, -1);
      n_index_fn(x) = select(flat_index != -1, flat_index / (w * h * c), -1);
      value_fn(x) = select(flat_index != -1, //
                           in_flattened(clamp(flat_index, 0, size - 1)),
                           Halide::cast<std::int8_t>(0));
      x_index_fn.compute_root();
      y_index_fn.compute_root();
      s_index_fn.compute_root();
      n_index_fn.compute_root();
      value_fn.compute_root();

      auto& x_indices = xysn[0];
      auto& y_indices = xysn[1];
      auto& s_indices = xysn[2];
      auto& n_indices = xysn[3];
      x_indices(x) = x_index_fn(x);
      y_indices(x) = y_index_fn(x);
      s_indices(x) = s_index_fn(x) + 1;  // Notice the +1 offset because we are processing SIFT.
      n_indices(x) = n_index_fn(x);
      value(x) = value_fn(x);

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        prefix_sum.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
        compacted_indices.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);

        x_index_fn.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
        y_index_fn.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
        s_index_fn.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
        n_index_fn.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
        value_fn.gpu_tile(x, xo, xi, tile_x, TailStrategy::GuardWithIf);
      }

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
