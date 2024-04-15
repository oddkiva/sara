#pragma once

#include <Eigen/Core>


namespace DO::Sara {

  template <typename T>
  class RgbColoredPoint
  {
  public:
    using Value = Eigen::Vector<T, 6>;
    using Coords = Eigen::Vector<T, 3>;
    using Color = Eigen::Vector<T, 3>;

    RgbColoredPoint() = default;

    RgbColoredPoint(const Value& v)
      : _v{v}
    {
    }

    inline operator Value&()
    {
      return _v;
    }

    inline operator const Value&() const
    {
      return _v;
    }

    inline auto coords() -> Eigen::Map<Coords>
    {
      return Eigen::Map<Eigen::Vector3<T>>{_v.data()};
    }

    inline auto coords() const -> Eigen::Map<const Coords>
    {
      return Eigen::Map<const Coords>{_v.data()};
    }

    inline auto color() -> Eigen::Map<Color>
    {
      return Eigen::Map<Color>{_v.data() + 3};
    }

    inline auto color() const -> Eigen::Map<const Color>
    {
      return Eigen::Map<const Color>{_v.data() + 3};
    }

  private:
    Value _v;
  };

}  // namespace DO::Sara
