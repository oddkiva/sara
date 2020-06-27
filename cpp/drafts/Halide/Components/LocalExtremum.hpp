
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


namespace DO::Shakti::HalideBackend {

  template <typename Input>
  auto local_max(const Input& f,                              //
                 const Halide::Var& x, const Halide::Var& y)  //
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    auto f_local_max = Halide::Func{f.name() + "local_max"};
    f_local_max(x, y) = Halide::maximum(f(x + r.x, y + r.y)) == f(x, y);
    return f;
  }

  template <typename Input>
  auto local_min(const Input& f,                              //
                 const Halide::Var& x, const Halide::Var& y)  //
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    auto f_local_min = Halide::Func{f.name() + "local_min"};
    f_local_min(x, y) = Halide::minimum(f(x + r.x, y + r.y)) == f(x, y);
    return f;
  }

  template <typename Input>
  auto local_scale_space_max(                                   //
      const Input& prev, const Input& curr, const Input& next,  //
      const Halide::Var& x, const Halide::Var& y)               //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return max(maximum(prev(x + r.x, y + r.y)),
               maximum(curr(x + r.x, y + r.y)),
               maximum(next(x + r.x, y + r.y))) == curr(x, y);
  }

  template <typename Input>
  auto local_scale_space_max(                                            //
      const Input& prev, const Input& curr, const Input& next,           //
      const Halide::Var& x, const Halide::Var& y, const Halide::Var& n)  //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return max(maximum(prev(x + r.x, y + r.y, n)),
               maximum(curr(x + r.x, y + r.y, n)),
               maximum(next(x + r.x, y + r.y, n))) == curr(x, y, n);
  }

  template <typename Input>
  auto local_scale_space_min(                                            //
      const Input& prev, const Input& curr, const Input& next,           //
      const Halide::Var& x, const Halide::Var& y, const Halide::Var& n)  //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return min(minimum(prev(x + r.x, y + r.y, n)),
               minimum(curr(x + r.x, y + r.y, n)),
               minimum(next(x + r.x, y + r.y, n))) == curr(x, y, n);
  }

}  // namespace DO::Shakti::HalideBackend
