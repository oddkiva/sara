#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO::Sara::MultipleObjectTracking {

  //! A pedestrian has a state which we model as a cylindric box and encode as a
  //! 4D vector
  //!   (x, y, a, h)
  //! where:
  //!   - (x, y) is the position of the pedestrian in meters w.r.t. the vehicle
  //!     frame
  //!   - a is the aspect ratio
  //!   - h is the height of the pedestrian
  //!
  //! In this modelling, we assume that the road is flat and on the plane
  //!   z = 0.
  //!
  //! That means that the 3D coordinates of the pedestrian feet is
  //!   (x, y, 0)
  //! in the vehicle frame.
  template <typename T>
  using CylindricBoxObservationVector = Eigen::Vector4<T>;

  //! In practice the pedestrian state is expanded into a larger 12D vector that
  //! concatenates the 3 4D vectors:
  //!   (x, y, a, h)         the base state vector
  //!   (dx, dy, da, dh)     1st-order differential of the base state vector
  //!   (d²x, d²y, d²a, d²h) 2nd-order differential of the base state vector
  //!
  //! The differentials are used indeed in the transition matrix in the Kalman
  //! filter.
  template <typename T>
  using CylindricBoxStateVector = Eigen::Vector<T, 12>;


  template <typename T>
  struct ObservationDistribution
  {
    using Scalar = T;
    using Mean = CylindricBoxObservationVector<T>;
    using CovarianceMatrix = Eigen::Matrix4<T>;

    auto mean() -> const Mean&
    {
      return μ;
    }

    auto covariance_matrix() -> const CovarianceMatrix&
    {
      return Σ;
    }

    Mean μ;
    CovarianceMatrix Σ;
  };

  template <typename T>
  struct StateDistribution
  {
    static constexpr auto dimension =
        CylindricBoxObservationVector<T>::RowsAtCompileTime;

    using Scalar = T;
    using Mean = CylindricBoxStateVector<T>;
    using CovarianceMatrix = Eigen::Matrix<T, dimension, dimension>;

    auto mean() -> const Mean&
    {
      return μ;
    }

    auto covariance_matrix() -> const CovarianceMatrix&
    {
      return Σ;
    }

    Mean μ;
    CovarianceMatrix Σ;
  };


}  // namespace DO::Sara::MultipleObjectTracking
