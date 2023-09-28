// ========================================================================== //
// this file is part of sara, a basic set of libraries in c++ for computer
// vision.
//
// copyright (c) 2023-present david ok <david.ok8@gmail.com>
//
// this source code form is subject to the terms of the mozilla public
// license v. 2.0. if a copy of the mpl was not distributed with this file,
// you can obtain one at http://mozilla.org/mpl/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  template <typename T, int D>
  struct PointList
  {
    using value_type = TensorView_<T, D - 1>;

    operator TensorView<T, D>&()
    {
      return _data;
    }

    operator const TensorView<T, D>&() const
    {
      return _data;
    }

    auto operator[](const int n) const -> value_type
    {
      return _data[n];
    }

    auto size() const -> int
    {
      return _data.size(0);
    }

    Tensor_<T, D> _data;
  };

}  // namespace DO::Sara
