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

#include <DO/Sara/Core/Tensor.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>


namespace DO::Sara {

  template <typename T>
  void print_3d_interleaved_int_array(const TensorView_<T, 3>& x)
  {
    using namespace std;

    const auto max = x.flat_array().abs().maxCoeff();
    std::stringstream ss;
    ss << max;
    const auto pad_size = ss.str().size();

    cout << "[";
    for (auto i = 0; i < x.size(0); ++i)
    {
      cout << "[";
      for (auto j = 0; j < x.size(1); ++j)
      {
        cout << "[";
        for (auto k = 0; k < x.size(2); ++k)
        {
          cout << std::setw(pad_size) << x(i, j, k);
          if (k != x.size(2) - 1)
            cout << ", ";
        }
        cout << "]";

        if (j != x.size(1) - 1)
          cout << ", ";
        else
          cout << "]";
      }

      if (i != x.size(0) - 1)
        cout << ",\n ";
    }
    cout << "]" << endl;
  }

  template <typename T>
  void print_3d_interleaved_float_array(const TensorView_<T, 3>& x)
  {
    using namespace std;

    cout << "[";
    for (auto i = 0; i < x.size(0); ++i)
    {
      cout << "[";
      for (auto j = 0; j < x.size(1); ++j)
      {
        cout << "[";
        for (auto k = 0; k < x.size(2); ++k)
        {
          cout << fixed << x(i, j, k);
          if (k != x.size(2) - 1)
            cout << ", ";
        }
        cout << "]";

        if (j != x.size(1) - 1)
          cout << ", ";
        else
          cout << "]";
      }

      if (i != x.size(0) - 1)
        cout << ",\n ";
    }
    cout << "]" << endl;
  }

  template <typename T>
  inline void print_3d_interleaved_array(const TensorView_<T, 3>& x)
  {
    if constexpr (std::numeric_limits<T>::is_integer)
      print_3d_interleaved_int_array(x);
    else
      print_3d_interleaved_float_array(x);
  }

}  // namespace DO::Sara
