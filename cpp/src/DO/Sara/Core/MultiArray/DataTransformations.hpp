// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>


namespace DO::Sara {

template <typename Predicate>
struct MultiArrayFilter
{
  Predicate predicate;

  inline MultiArrayFilter(Predicate p)
    : predicate{p}
  {
  }

  template <typename T, int O>
  auto operator()(const MultiArrayView<T, 1, O>& v)
  {
    auto v_copy = MultiArray<T, 1, O>{v.sizes()};
    const auto v_end =
        std::copy_if(std::begin(v), std::end(v), std::begin(v_copy), predicate);
    const auto v_begin = std::begin(v_copy);

    auto v_filtered = MultiArray<T, 1, O>{static_cast<int>(v_end - v_begin)};
    std::copy(v_begin, v_end, std::begin(v_filtered));

    return v_filtered;
  };

  template <typename T, int O>
  auto operator()(MultiArray<T, 1, O>&& v)
  {
    const auto v_end =
        std::copy_if(std::begin(v), std::end(v), std::begin(v), predicate);
    const auto v_begin = std::begin(v);

    auto v_filtered = MultiArray<T, 1, O>{static_cast<int>(v_end - v_begin)};
    std::copy(v_begin, v_end, std::begin(v_filtered));

    return v_filtered;
  };
};


template <typename Op>
struct MultiArrayTransform
{
  Op op;

  inline MultiArrayTransform(Op p)
    : op{p}
  {
  }

  template <typename T, int N, int O>
  auto operator()(const MultiArrayView<T, N, O>& v)
  {
    using U = decltype(op(std::declval<T>()));
    auto res = MultiArray<U, N, O>{v.sizes()};
    std::transform(std::begin(v), std::end(v), std::begin(res), op);
    return res;
  };
};


template <typename Predicate>
auto filtered(Predicate p)
{
  return MultiArrayFilter<Predicate>{p};
}

template <typename Op>
inline auto transformed(Op op)
{
  return MultiArrayTransform<Op>{op};
}


template <typename T, int N, int O, typename Op>
inline auto operator|(const MultiArrayView<T, N, O>& t, Op op)
{
  return op(t);
}

} /* namespace DO::Sara */
