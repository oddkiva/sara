#pragma once


namespace DO { namespace Sara {

  namespace runtime {

    template <typename T>
    class TensorView
    {
    public:
      DynamicTensorView() = default;

      DynamicTensorView& reshape(const std::vector<std::size_t>& shape);

    public:
      T* data{nullptr};
      std::vector<std::size_t> shape;
      std::vector<std::size_t> strides;
    };

    template <typename T>
    class Tensor
    {
    public:
      std::unique_ptr<T[]> data;
      std::vector<std::size_t> shape;
      std::vector<std::size_t> strides;
    };

  }


  auto patch(ImageView<T, 3> in, const Vector3i& beg, const Vector3i& end);
  auto vec(auto x);

  template <typename T>
  MatrixXf im2col(ImageView<T, 3> in, int kw, int kh, int kc)
  {
    const auto w = in.width();
    const auto h = in.height();
    auto phi = MatrixXf{kw * kh * kc, w*h};

    for (auto c = 0u; c < w*h; ++c)
    {
      auto x = c % w;
      auto y = c / w;
      auto s = Vector3i{x - kw / 2, y - kh / 2, k - kc / 2};
      auto e = Vector3i{x + kw / 2, y + kh / 2, k + kc / 2};
      phi.col(c) = vec(patch(in, s, e));
    }
  }

} /* namespace Sara */
} /* namespace DO */
