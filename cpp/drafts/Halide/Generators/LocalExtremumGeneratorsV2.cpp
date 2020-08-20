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

#include <drafts/Halide/Components/LocalExtremum.hpp>
#include <drafts/Halide/Components/DoGExtremum.hpp>


namespace v2 {

  using namespace Halide;


  template <typename T>
  class LocalMax : public Generator<LocalMax<T>>
  {
    using Base = Generator<LocalMax<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

  public:
    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> scale_prev{"prev", 4};
    Input<Buffer<T>> scale_curr{"curr", 4};
    Input<Buffer<T>> scale_next{"next", 4};

    Output<Buffer<std::uint8_t>> out{"out", 3};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    void generate()
    {
      const auto prev_ext = BoundaryConditions::repeat_edge(scale_prev);
      const auto curr_ext = BoundaryConditions::repeat_edge(scale_curr);
      const auto next_ext = BoundaryConditions::repeat_edge(scale_next);

      using DO::Shakti::HalideBackend::local_scale_space_max;
      out(x, y, c, n) = cast<std::uint8_t>(
          local_scale_space_max(prev_ext, curr_ext, next_ext, x, y, c, n));
    }

    void schedule()
    {
      auto& prev = scale_prev;
      auto& curr = scale_curr;
      auto& next = scale_next;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        out.gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1,
                      TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        out.hexagon()
            .prefetch(prev, y, 2)
            .prefetch(curr, y, 2)
            .prefetch(next, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        out.split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

  template <typename T>
  class LocalScaleSpaceExtremum : public Generator<LocalScaleSpaceExtremum<T>>
  {
    using Base = Generator<LocalScaleSpaceExtremum<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

  public:
    GeneratorParam<int> tile_x{"tile_x", 16};
    GeneratorParam<int> tile_y{"tile_y", 16};

    Input<Buffer<T>> scale_prev{"prev", 4};
    Input<Buffer<T>> scale_curr{"curr", 4};
    Input<Buffer<T>> scale_next{"next", 4};
    Input<T> edge_ratio{"edge_ratio"};
    Input<T> extremum_thres{"extremum_thres"};

    Output<Buffer<std::int8_t>> out{"out", 4};

    //! @brief Variables.
    //! @{
    Var x{"x"}, y{"y"}, c{"c"}, n{"n"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};
    //! @}

    void generate()
    {
      const auto prev_ext = BoundaryConditions::repeat_edge(scale_prev);
      const auto curr_ext = BoundaryConditions::repeat_edge(scale_curr);
      const auto next_ext = BoundaryConditions::repeat_edge(scale_next);

      using DO::Shakti::HalideBackend::is_dog_extremum;
      out(x, y, c, n) = is_dog_extremum(prev_ext, curr_ext, next_ext,  //
                                        edge_ratio, extremum_thres,    //
                                        x, y, c, n);                   //
    }

    void schedule()
    {
      auto& prev = scale_prev;
      auto& curr = scale_curr;
      auto& next = scale_next;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        out.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y,
                     TailStrategy::GuardWithIf);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        out.hexagon()
            .prefetch(prev, y, 2)
            .prefetch(curr, y, 2)
            .prefetch(next, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size, TailStrategy::GuardWithIf);
      }

      // CPU schedule.
      else
      {
        out.split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8, TailStrategy::GuardWithIf);
      }
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(v2::LocalMax<float>, shakti_local_max_32f_v2)
HALIDE_REGISTER_GENERATOR(v2::LocalScaleSpaceExtremum<float>,
                          shakti_local_scale_space_extremum_32f_v2)
