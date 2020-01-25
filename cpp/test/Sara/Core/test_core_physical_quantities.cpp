#include <DO/Sara/Core/PhysicalQuantities.hpp>


namespace sara = DO::Sara;


auto main() -> int
{
  // Area dimensional check
  static_assert(
      std::is_same_v<sara::Area, decltype(sara::Length{} * sara::Length{})>);
  static_assert(
      std::is_same_v<
          sara::Area,
          sara::PhysicalQuantity<std::ratio<0>, std::ratio<2>, std::ratio<0>>>);


  // Volume dimensional check
  static_assert(
      std::is_same_v<sara::Volume, decltype(sara::Length{} * sara::Length{} *
                                            sara::Length{})>);

  static_assert(
      std::is_same_v<
          sara::Volume,
          sara::PhysicalQuantity<std::ratio<0>, std::ratio<3>, std::ratio<0>>>);


  // Speed.
  static_assert(
      std::is_same_v<sara::Speed,
                     sara::PhysicalQuantity<std::ratio<0>, std::ratio<1>,
                                            std::ratio<-1>>>);

  //
  static_assert(
      std::is_same_v<sara::AccelerationScalar,
                     sara::PhysicalQuantity<std::ratio<0>, std::ratio<1>,
                                            std::ratio<-2>>>);

  //
  static_assert(
      std::is_same_v<sara::ForceScalar,
                     sara::PhysicalQuantity<std::ratio<1>, std::ratio<1>,
                                            std::ratio<-2>>>);

  static_assert(
      std::is_same_v<sara::Pressure,
                     sara::PhysicalQuantity<std::ratio<1>, std::ratio<-1>,
                                            std::ratio<-2>>>);
  return 0;
}
