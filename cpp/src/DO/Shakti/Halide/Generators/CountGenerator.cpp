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

#include <DO/Shakti/Halide/Components/DoGExtremum.hpp>
#include <DO/Shakti/Halide/Components/LocalExtremum.hpp>


namespace {

  using namespace Halide;

  template <typename T>
  class CountExtrema : public Generator<CountExtrema<T>>
  {
    using Base = Generator<CountExtrema<T>>;
    using Base::get_target;

    template <typename T2>
    using Input = typename Base::template Input<T2>;

    template <typename T2>
    using Output = typename Base::template Output<T2>;

  public:
    GeneratorParam<int> tile_x{"tile_x", 8};
    GeneratorParam<int> tile_y{"tile_y", 8};
    GeneratorParam<int> tile_s{"tile_s", 4};

    Input<Buffer<T>> f{"f", 4};
    Output<std::int32_t> out{"count"};

    //! @brief Variables.
    //! @{
    RVar x{"x"}, n{"n"};
    RVar xo{"xo"}, yo{"yo"}, so{"so"};
    RVar xi{"xi"}, yi{"yi"}, si{"si"};
    //! @}

    Var u{"u"}, v{"v"}, w{"w"};

    void generate()
    {
      const auto& w = f.dim(0).extent();
      const auto& h = f.dim(1).extent();
      const auto& c = f.dim(2).extent();
      const auto& n = f.dim(3).extent();

      auto r = RDom(0, w, 0, h, 0, c, 0, n);

      auto nonzero = select(           //
          f(r.x, r.y, r.z, r.w) != 0,  //
          std::int32_t{1},             //
          std::int32_t{0});
      out() = sum(nonzero);

      // out.update()
      //     .split(r.x, xo, xi, tile_x, TailStrategy::GuardWithIf)
      //     .split(r.y, yo, yi, tile_y, TailStrategy::GuardWithIf);
      //     //.split(r.z, so, si, tile_s, TailStrategy::GuardWithIf);

      // // We now call rfactor to make an intermediate function that
      // // independently computes a histogram of each tile.
      // Func intermediate = out.update().rfactor({{xo, u}, {yo, v}});

      // // We can now parallelize the intermediate over tiles.
      // intermediate.compute_root().update().parallel(u).parallel(v);

      // // We also reorder the tile indices outermost to give the
      // // classic tiled traversal.
      // intermediate.update().reorder(xi, yi, u, v);


      // Func sum_val;
      // sum_val() = 0;
      // sum_val() += nonzero;
      // sum_val.update().split(r.x, xo, xi, 4).split(r.y, yo, yi, 4);

      // Func intermediate = sum_val.update().rfactor({{xo, u}, {yo, v}});
      // intermediate.compute_root().update().parallel(u).parallel(v);

      // out() = sum_val();
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(CountExtrema<std::int8_t>, shakti_count_extrema)
