#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  template <typename T>
  struct PointCorrespondenceList
  {
    using value_type = std::array<const TensorView_<T, 1>, 2>;

    PointCorrespondenceList() = default;

    PointCorrespondenceList(const TensorView_<int, 2>& M,
                            const TensorView_<T, 2>& p1,
                            const TensorView_<T, 2>& p2)
      : _p1{M.size(0), p1.size(1)}
      , _p2{M.size(0), p2.size(1)}
    {
      auto p1_mat = p1.matrix();
      auto p2_mat = p2.matrix();
      auto p1_matched = _p1.matrix();
      auto p2_matched = _p2.matrix();
      for (auto m = 0; m < M.size(0); ++m)
      {
        const auto& i1 = M(m, 0);
        const auto& i2 = M(m, 1);

        p1_matched.row(m) = p1_mat.row(i1);
        p2_matched.row(m) = p2_mat.row(i2);
      }
    }

    auto size() const -> int
    {
      return _p1.size(0);
    }

    auto operator[](const int n) const -> value_type
    {
      return {_p1[n], _p2[n]};
    }

    Tensor_<T, 2> _p1;
    Tensor_<T, 2> _p2;
  };

  template <typename T>
  struct PointCorrespondenceSubsetList
  {
    using value_type = std::array<TensorView_<T, 2>, 2>;

    auto operator[](const int n) const -> value_type
    {
      return {_p1[n], _p2[n]};
    }

    Tensor_<T, 3> _p1;
    Tensor_<T, 3> _p2;
  };

}  // namespace DO::Sara
