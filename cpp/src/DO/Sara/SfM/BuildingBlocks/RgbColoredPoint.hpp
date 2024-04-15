#pragma once

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T>
  struct RgbColoredPoint
  {
    using Value = Eigen::Vector<T, 6>;
    using Coords = Eigen::Vector<T, 3>;
    using Color = Eigen::Vector<T, 3>;

    inline operator Value&()
    {
      return value;
    }

    inline operator const Value&() const
    {
      return value;
    }

    inline auto coords() -> Eigen::Map<Coords>
    {
      return Eigen::Map<Eigen::Vector3<T>>{value.data()};
    }

    inline auto coords() const -> Eigen::Map<const Coords>
    {
      return Eigen::Map<const Coords>{value.data()};
    }

    inline auto color() -> Eigen::Map<Color>
    {
      return Eigen::Map<Color>{value.data() + 3};
    }

    inline auto color() const -> Eigen::Map<const Color>
    {
      return Eigen::Map<const Color>{value.data() + 3};
    }

  private:
    Value value;
  };

}  // namespace DO::Sara
