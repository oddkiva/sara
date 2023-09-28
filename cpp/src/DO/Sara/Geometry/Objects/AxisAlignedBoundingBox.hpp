#pragma once

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T, int N>
  struct AxisAlignedBoundingBox
  {
    auto start() -> Eigen::Map<Eigen::Vector<T, N>>
    {
      return Eigen::Map<Eigen::Vector<T, N>>{_data.data(), N};
    }

    auto start() const -> Eigen::Map<const Eigen::Vector<T, N>>
    {
      return Eigen::Map<const Eigen::Vector<T, N>>{_data.data(), N};
    }

    auto end() const -> Eigen::Vector<T, N>
    {
      return start() + sizes();
    }

    auto sizes() -> Eigen::Map<Eigen::Vector<T, N>>
    {
      return Eigen::Map<Eigen::Vector<T, N>>{_data.data() + N, N};
    }

    auto sizes() const -> Eigen::Map<const Eigen::Vector<T, N>>
    {
      return Eigen::Map<const Eigen::Vector<T, N>>{_data.data() + N, N};
    }

    auto top_left() -> Eigen::Map<Eigen::Vector<T, N>>
    {
      static_assert(N == 2);
      return start();
    }

    auto top_left() const -> Eigen::Map<const Eigen::Vector<T, N>>
    {
      static_assert(N == 2);
      return start();
    }

    auto bottom_right() const -> Eigen::Map<const Eigen::Vector<T, N>>
    {
      static_assert(N == 2);
      return end();
    }

    auto x() -> T&
    {
      return _data(0);
    }

    auto x() const -> const T&
    {
      return _data(0);
    }

    auto y() -> T&
    {
      return _data(1);
    }

    auto y() const -> const T&
    {
      return _data(1);
    }

    auto width() -> T&
    {
      return _data(N);
    }

    auto width() const -> const T&
    {
      return _data(N);
    }

    auto height() -> T&
    {
      return _data(N + 1);
    }

    auto height() const -> const T&
    {
      return _data(N + 1);
    }

    Eigen::Vector<T, N * 2> _data;
  };

}  // namespace DO::Sara
