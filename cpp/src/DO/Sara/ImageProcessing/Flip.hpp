#pragma once

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  enum FlipType
  {
    Horizontal = 0,
    Vertical = 1,
  };

  template <typename T>
  void flip(ImageView<T>& image, FlipType flip_type)
  {
    if (flip_type == Horizontal)
    {
      const auto N = image.cols();
      auto m = image.matrix();
      for (int j = 0; j < N / 2; ++j)
        m.col(j).swap(m.col(N - j));
    }

    if (flip_type == Vertical)
    {
      const auto M = image.rows();
      auto m = image.matrix();
      for (int i = 0; i < M / 2; ++i)
        m.row(i).swap(m.row(M - i));
    }
  }

} /* namespace Sara */
} /* namespace DO */
