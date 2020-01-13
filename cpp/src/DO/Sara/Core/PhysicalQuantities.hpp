#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif

#include <Eigen/Core>

#include <cmath>
#include <ratio>
#include <utility>


namespace DO::Sara {

  //! @ingroup Core
  //! @defgroup Physics Physics
  //! @{

  //! @brief Generic physical quantity
  template <typename T, typename... Q>
  struct QuantityBase
  {
    using exponents_type = std::tuple<Q...>;

    using scalar_type = T;

    inline constexpr QuantityBase() = default;

    inline explicit constexpr QuantityBase(scalar_type v)
      : value{v}
    {
    }

    inline explicit constexpr operator const scalar_type() const
    {
      return value;
    }

    inline explicit operator scalar_type&()
    {
      return value;
    }

    inline constexpr auto operator+(const QuantityBase& other) const
    {
      return QuantityBase{value + other.value};
    }

    inline constexpr auto operator-(const QuantityBase& other) const
    {
      return QuantityBase{value + other.value};
    }

    inline constexpr auto operator*(scalar_type scale) const -> QuantityBase
    {
      return QuantityBase{value * scale};
    }

    inline constexpr auto operator/(scalar_type scale) const -> QuantityBase
    {
      return QuantityBase{value / scale};
    }

    inline auto operator+=(const QuantityBase& other) -> QuantityBase&
    {
      value += other.value;
      return *this;
    }

    inline auto operator-=(const QuantityBase& other) -> QuantityBase
    {
      value -= other.value;
      return *this;
    }

    inline auto operator*=(scalar_type scale) -> QuantityBase&
    {
      value *= scale;
      return *this;
    }

    inline auto operator/=(scalar_type scale) -> QuantityBase&
    {
      value /= scale;
      return *this;
    }

    T value{};
  };

  template <typename T, typename... Q>
  inline constexpr auto operator*(T scale, const QuantityBase<T, Q...>& q)
  {
    return QuantityBase<T, Q...>{scale * q.value};
  }


  //! @brief Exponent alias.
  template <std::intmax_t P, std::intmax_t Q = 1>
  using Exp = std::ratio<P, Q>;


  //! @brief Physical quantity.
  template <typename MassExp,                               //
            typename LengthExp,                             //
            typename TimeExp,                               //
            typename ElectricCurrentExp = Exp<0>,           //
            typename ThermodynamicTemperatureExp = Exp<0>,  //
            typename AmountOfSubstanceExp = Exp<0>,         //
            typename LuminousIntensityExp = Exp<0>,         //
            typename T = long double>
  using PhysicalQuantity = QuantityBase<T,                            //
                                        MassExp,                      //
                                        LengthExp,                    //
                                        TimeExp,                      //
                                        ElectricCurrentExp,           //
                                        ThermodynamicTemperatureExp,  //
                                        AmountOfSubstanceExp,         //
                                        LuminousIntensityExp>;


  template <typename m1, typename l1, typename t1,                //
            typename I1, typename T1, typename n1, typename Iv1,  //
            typename m2, typename l2, typename t2,                //
            typename I2, typename T2, typename n2, typename Iv2,  //
            typename T>
  inline constexpr auto
  operator*(const PhysicalQuantity<m1, l1, t1, I1, T1, n1, Iv1, T>& q1,
            const PhysicalQuantity<m2, l2, t2, I2, T2, n2, Iv2, T>& q2)
  {
    return PhysicalQuantity<std::ratio_add<m1, m2>,    // mass
                            std::ratio_add<l1, l2>,    // length
                            std::ratio_add<t1, t2>,    // time
                            std::ratio_add<I1, I2>,    // electric current
                            std::ratio_add<T1, T2>,    // temperature
                            std::ratio_add<n1, n2>,    // amount of substance
                            std::ratio_add<Iv1, Iv2>,  // luminous intensity
                            T>                         // scalar type
        {q1.value * q2.value};
  }

  template <typename m1, typename l1, typename t1,                //
            typename I1, typename T1, typename n1, typename Iv1,  //
            typename m2, typename l2, typename t2,                //
            typename I2, typename T2, typename n2, typename Iv2,  //
            typename T>
  inline constexpr auto
  operator/(const PhysicalQuantity<m1, l1, t1, I1, T1, n1, Iv1, T>& q1,
            const PhysicalQuantity<m2, l2, t2, I2, T2, n2, Iv2, T>& q2)
  {
    return PhysicalQuantity<std::ratio_subtract<m1, m2>,  // mass
                            std::ratio_subtract<l1, l2>,  // length
                            std::ratio_subtract<t1, t2>,  // time
                            std::ratio_subtract<I1, I2>,  // electric current
                            std::ratio_subtract<T1, T2>,  // temperature
                            std::ratio_subtract<n1, n2>,  // amount of substance
                            std::ratio_subtract<Iv1, Iv2>,  // luminous
                            // intensity
                            T>{q1.value * q2.value};
  }


  template <typename T, typename Exp, typename... Q>
  auto pow(QuantityBase<T, Q...> q, Exp)
      -> QuantityBase<T, std::ratio_multiply<Q, Exp>...>
  {
    return {std::pow(q.value, static_cast<long double>(Exp::num) / Exp::den)};
  }


  //! @brief Define a number as a mechanical quantity with zero exponents
  //! everywhere.
  /*
   *  We limit the scope to mechanical quantity and ignores
   */
  using Number = PhysicalQuantity<Exp<0>, Exp<0>, Exp<0>>;


  //! @brief Usual physical quantities.
  using Mass = PhysicalQuantity<Exp<1>, Exp<0>, Exp<0>>;
  using Length = PhysicalQuantity<Exp<0>, Exp<1>, Exp<0>>;
  using Time = PhysicalQuantity<Exp<0>, Exp<0>, Exp<1>>;

  using ElectricCurrent = PhysicalQuantity<Exp<0>, Exp<0>, Exp<0>, Exp<1>>;
  using Temperature = PhysicalQuantity<Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<1>>;
  using AmountOfSubstance =
      PhysicalQuantity<Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<1>>;
  using LuminousIntensity =
      PhysicalQuantity<Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<0>, Exp<1>>;

  using Area = decltype(Length{} * Length{});
  using Volume = decltype(Length{} * Length{} * Length{});

  using Speed = decltype(Length{} / Time{});
  using AccelerationScalar = decltype(Speed{} / Time{});
  using ForceScalar = decltype(Mass{} * AccelerationScalar{});

  using Pressure = decltype(ForceScalar{} / Area{});

  using Energy = decltype(Mass{} * Speed{} * Speed{});
  using Work = decltype(ForceScalar{} * Length{});


  using Frequency = decltype(Number{} / Time{});


  // Physical vector, tensors.
  template <int N>
  using Position = Eigen::Matrix<Length, N, 1>;

  template <int N>
  using Sizes = Eigen::Matrix<Length, N, 1>;

  template <int N>
  using Velocity = Eigen::Matrix<Speed, N, 1>;

  template <int N = 1>
  using Acceleration = Eigen::Matrix<AccelerationScalar, N, 1>;

  template <int N>
  using Force = Eigen::Matrix<ForceScalar, N, 1>;

  template <int N>
  using StressTensor = Eigen::Matrix<Pressure, N, N>;


  // Dimension-less quantity.
  template <typename Derived>
  struct DimensionlessQuantity : Number
  {
    using base_type = Number;
    using self_type = DimensionlessQuantity;
    using derived_type = Derived;

    using scalar_type = Number::scalar_type;

    inline constexpr DimensionlessQuantity() = default;

    inline constexpr DimensionlessQuantity(const base_type& other)
      : base_type{other}
    {
    }

    inline constexpr DimensionlessQuantity(base_type&& other)
      : base_type{other}
    {
    }

    inline constexpr DimensionlessQuantity(long double v)
      : Number{v}
    {
    }

    auto operator=(const self_type&) -> self_type& = default;
  };


#define MAKE_DIMENSIONLESS_QUANTITY(ClassName)                                 \
  struct ClassName : DimensionlessQuantity<ClassName>                          \
  {                                                                            \
    using base_type = DimensionlessQuantity<ClassName>;                        \
                                                                               \
    inline constexpr ClassName() = default;                                    \
                                                                               \
    inline constexpr ClassName(const base_type& v)                             \
      : base_type{v}                                                           \
    {                                                                          \
    }                                                                          \
                                                                               \
    inline constexpr ClassName(long double v)                                  \
      : base_type{v}                                                           \
    {                                                                          \
    }                                                                          \
                                                                               \
    auto operator=(const ClassName&) -> ClassName& = default;                  \
  }

  // Geometric quantities.
  MAKE_DIMENSIONLESS_QUANTITY(Angle);
  MAKE_DIMENSIONLESS_QUANTITY(SolidAngle);


  // Some SI units.
  constexpr auto kilogram = Mass{1};

  constexpr auto meter = Length{1};
  constexpr auto kilometer = meter * 1000;
  constexpr auto centimeter = meter / 100;
  constexpr auto millimeter = meter / 1000;

  constexpr auto foot = Length{0.3048};
  constexpr auto inch = Length{0.0254};

  constexpr auto second = Time{1};

  constexpr auto hertz = Frequency{1};

  constexpr auto radian = Angle{1};
  constexpr auto degree = radian * 180.L / M_PI;


  // User-defined literal operators.
  constexpr auto operator""_kg(long double v)
  {
    return v * kilogram;
  }


  constexpr auto operator""_km(long double v)
  {
    return v * kilometer;
  }

  constexpr auto operator""_m(long double v)
  {
    return v * meter;
  }

  constexpr auto operator""_cm(long double v)
  {
    return v * centimeter;
  }

  constexpr auto operator""_mm(long double v)
  {
    return v * millimeter;
  }


  constexpr auto operator""_s(long double v)
  {
    return v * second;
  }

  constexpr auto operator""_Hz(long double v)
  {
    return v * hertz;
  }


  constexpr auto operator""_radian(long double v) -> Angle
  {
    return {v * radian.value};
  }

  constexpr auto operator""_degree(long double v) -> Angle
  {
    return {v * degree.value};
  }

  //! @} Physics

}  // namespace DO::Sara


// Computer vision units.
namespace DO::Sara {

  //! @addtogroup Physics
  //! @{

  // Image quantities.
  MAKE_DIMENSIONLESS_QUANTITY(PixelUnit);

  using PixelUnitPerLength = decltype(PixelUnit{} / Length{});

  template <int N>
  using PixelUnits = Eigen::Matrix<PixelUnit, N, 1>;

  template <int N>
  using PixelUnitsPerLength = Eigen::Matrix<PixelUnitPerLength, N, 1>;


  constexpr auto operator""_px(long double v) -> PixelUnit
  {
    return {v};
  }

  constexpr auto operator""_fps(long double v)
  {
    return v * hertz;
  }


  // Only valid for the pinhole camera model and when there is not distortion
  // and no shear component.
  auto pixel_sizes(double u, double v) -> Sizes<2>
  {
    return {u, v};
  }


  // Sizes = [1920, 1080] -> 16/9
  auto aspect_ratio(Sizes<2> sensor_sizes)
  {
    const auto& sensor_width = sensor_sizes[0];
    const auto& sensor_height = sensor_sizes[0];
    return sensor_width / sensor_height;
  }

  auto pixels_per_length(const PixelUnits<2>& image_sizes,
                         const Sizes<2>& sensor_sizes) -> PixelUnitsPerLength<2>
  {
    return image_sizes.cast<long double>()
        .cwiseQuotient(sensor_sizes.cast<long double>())
        .cast<PixelUnitPerLength>();
  }


  // N.B.: this is true only if pixels are square!
  auto focal_lengths_in_pixels(const PixelUnits<2>& image_sizes,  //
                               const Sizes<2>& sensor_sizes,      //
                               Length focal_length)               //
      -> PixelUnits<2>
  {
    const auto ratio = sensor_sizes.cast<long double>() / focal_length.value;
    return image_sizes.cast<long double>().cwiseProduct(ratio).cast<PixelUnit>();
  }

}  // namespace DO::Sara
