
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

#pragma once

#include <drafts/Halide/MyHalide.hpp>


namespace DO { namespace Shakti { namespace HalideBackend {

  //! @brief Local extremum predicate in Halide.
  //! @{
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
  auto local_scale_space_min(                                   //
      const Input& prev, const Input& curr, const Input& next,  //
      const Halide::Var& x, const Halide::Var& y)               //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return min(minimum(prev(x + r.x, y + r.y)),
               minimum(curr(x + r.x, y + r.y)),
               minimum(next(x + r.x, y + r.y))) == curr(x, y);
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
  //! @}

  //! @brief Local extremum predicate in Halide.
  /*!
   *
   *  Same implementation to support 4D tensors.
   *  This is by design to support seamless interoperability GPU AOT
   *  computation.
   *
   *  In practice we always have c == 1 and n == 1.
   *
   */
  //! @{
  template <typename Input>
  auto local_max(const Input& f,                              //
                 const Halide::Var& x, const Halide::Var& y,  //
                 const Halide::Var& c, const Halide::Var& n)  //
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    auto f_local_max = Halide::Func{f.name() + "local_max"};
    f_local_max(x, y, c, n) =
        Halide::maximum(f(x + r.x, y + r.y), c, n) == f(x, y, c, n);
    return f;
  }

  template <typename Input>
  auto local_min(const Input& f,                              //
                 const Halide::Var& x, const Halide::Var& y,  //
                 const Halide::Var& c, const Halide::Var& n)  //
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    auto f_local_min = Halide::Func{f.name() + "local_min"};
    f_local_min(x, y, c, n) =
        Halide::minimum(f(x + r.x, y + r.y), c, n) == f(x, y, c, n);
    return f;
  }

  template <typename Input>
  auto local_scale_space_max(                                   //
      const Input& prev, const Input& curr, const Input& next,  //
      const Halide::Var& x, const Halide::Var& y,               //
      const Halide::Var& c, const Halide::Var& n)               //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return max(maximum(prev(x + r.x, y + r.y, c, n)),
               maximum(curr(x + r.x, y + r.y, c, n)),
               maximum(next(x + r.x, y + r.y, c, n))) == curr(x, y, c, n);
  }

  template <typename Input>
  auto local_scale_space_min(                                   //
      const Input& prev, const Input& curr, const Input& next,  //
      const Halide::Var& x, const Halide::Var& y,               //
      const Halide::Var& c, const Halide::Var& n)               //
  {
    using namespace Halide;
    const auto r = RDom{-1, 3, -1, 3};
    return min(minimum(prev(x + r.x, y + r.y, c, n)),
               minimum(curr(x + r.x, y + r.y, c, n)),
               minimum(next(x + r.x, y + r.y, c, n))) == curr(x, y, c, n);
  }

  template <typename Input>
  auto local_max_4d(const Input& f,                //
                    const Halide::Expr& x_min,     //
                    const Halide::Expr& y_min,     //
                    const Halide::Expr& c_min,     //
                    const Halide::Expr& n_min,     //
                    const Halide::Expr& x_extent,  //
                    const Halide::Expr& y_extent,  //
                    const Halide::Expr& c_extent,  //
                    const Halide::Expr& n_extent,  //
                    const Halide::Var& x,          //
                    const Halide::Var& y,          //
                    const Halide::Var& c,          //
                    const Halide::Var& n)          //
  {
    auto r = Halide::RDom{x_min, x_extent,  //
                          y_min, y_extent,  //
                          c_min, c_extent,  //
                          n_min, n_extent};
    return Halide::maximum(f(x + r.x, y + r.y, c + r.z, n + r.w)) ==
           f(x, y, c, n);
  }

  template <typename Input>
  auto local_min_4d(const Input& f,                //
                    const Halide::Expr& x_min,     //
                    const Halide::Expr& y_min,     //
                    const Halide::Expr& c_min,     //
                    const Halide::Expr& n_min,     //
                    const Halide::Expr& x_extent,  //
                    const Halide::Expr& y_extent,  //
                    const Halide::Expr& c_extent,  //
                    const Halide::Expr& n_extent,  //
                    const Halide::Var& x,          //
                    const Halide::Var& y,          //
                    const Halide::Var& c,          //
                    const Halide::Var& n)          //
  {
    auto r = Halide::RDom{x_min, x_extent,  //
                          y_min, y_extent,  //
                          c_min, c_extent,  //
                          n_min, n_extent};
    return Halide::min(f(x + r.x, y + r.y, c + r.z, n + r.w)) == f(x, y, c, n);
  }

  //! @}

}}}  // namespace DO::Shakti::HalideBackend
