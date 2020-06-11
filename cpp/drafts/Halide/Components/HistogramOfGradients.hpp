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

#include <drafts/Halide/MyHalide.hpp>


namespace {

  using namespace Halide;

  struct HistogramOfGradients
  {
    //! @brief Variables.
    Var x{"x"}, y{"y"}, o{"o"};

    template <typename Gradients, typename HoG>
    void generate(Gradients& gradients, HoG& hog,
                  Expr w_in, Expr h_in,
                  Expr wb, Expr hb, Expr O)
    {
      RDom r{-wb / 2, wb, -hb / 2, hb};
      auto mag = select(                       //
          (0 <= x + r.x && x + r.x < w_in) &&  //
          (0 <= y + r.y && y + r.y < h_in),
          gradients(x + r.x, y + r.y, 0), 0);
      auto ori = select(                       //
          (0 <= x + r.x && x + r.x < w_in) &&  //
          (0 <= y + r.y && y + r.y < h_in),
          gradients(x + r.x, y + r.y, 1), 0);
      auto ori_normalized = (ori < 0 ? ori + (2 * M_PI) : ori) / (2 * M_PI);
      auto ori_index = ori_normalized * O;

      auto dx = abs(2 * (x - r.x) / wb);
      auto dy = abs(2 * (y - r.y) / hb);
      auto dori = abs(ori_index - o);
      auto wx = select(dx < 1, 1 - dx , 0);
      auto wy = select(dy < 1, 1 - dy , 0);
      auto wo = select(dori < 1, (1 - dori, 0);
      hog(x, y, o) = sum(wx * wy * wo * mag);
    }
  };

}  // namespace
