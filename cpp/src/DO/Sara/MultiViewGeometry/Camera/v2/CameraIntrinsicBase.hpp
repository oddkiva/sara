#pragma once

#include <Eigen/Core>

#include <type_traits>


namespace DO::Sara::v2 {

  template <typename T>
  concept ArrayConcept = requires
  {
    std::is_array_v<T>;
    typename T::value_type;
  };

  template <ArrayConcept Array>
  struct CameraIntrinsicBase
  {
    using array_type = Array;
    Array data;
  };


  template <typename T, int N>
  using VectorView = Eigen::Map<Eigen::Vector<T, N>>;

  template <typename T, int N>
  using ConstVectorView = Eigen::Map<const Eigen::Vector<T, N>>;


  template <typename T, int N>
  using CameraIntrinsicSpan = CameraIntrinsicBase<VectorView<T, N>>;

  template <typename T, int N>
  using CameraIntrinsicVector = CameraIntrinsicBase<Eigen::Vector<T, N>>;

}  // namespace DO::Sara::v2
