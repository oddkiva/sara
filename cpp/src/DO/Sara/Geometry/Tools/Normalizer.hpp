// ========================================================================== //
// this file is part of sara, a basic set of libraries in c++ for computer
// vision.
//
// copyright (c) 2023-present david ok <david.ok8@gmail.com>
//
// this source code form is subject to the terms of the mozilla public
// license v. 2.0. if a copy of the mpl was not distributed with this file,
// you can obtain one at http://mozilla.org/mpl/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Geometry/Algorithms/RobustEstimation/PointList.hpp>
#include <DO/Sara/Geometry/Tools/Projective.hpp>


namespace DO::Sara {

  //! @ingroup Geometry
  //! @defgroup GeometryDataNormalizer Normalizers
  //! @{

  template <typename S>
  inline auto compute_normalizer(const TensorView_<S, 2>& X) -> Matrix<S, 3, 3>
  {
    auto slope = Eigen::RowVector2<S>{};
    auto offset = Eigen::RowVector2<S>{};

    if (X.size(1) == 2)
    {
      const Eigen::RowVector2<S> min = X.matrix().colwise().minCoeff();
      const Eigen::RowVector2<S> max = X.matrix().colwise().maxCoeff();

      slope = (max - min).cwiseInverse();
      offset = -min.cwiseQuotient(max - min);
    }
    else if (X.size(1) == 3)
    {
      const Eigen::RowVector3<S> min = X.matrix().colwise().minCoeff();
      const Eigen::RowVector3<S> max = X.matrix().colwise().maxCoeff();

      slope = (max - min).cwiseInverse().head(2);
      offset = -min.cwiseQuotient(max - min).head(2);
    }
    else
      throw std::runtime_error{
          "To compute the normalization matrix the input data "
          "dimension must be 2 or 3!"};


    auto T = Eigen::Matrix<S, 3, 3>{};
    T.setZero();
    T.template topLeftCorner<2, 2>() = slope.asDiagonal();
    T.col(2) << offset.transpose(), S(1);

    return T;
  }

  template <typename S>
  inline auto apply_transform(const Matrix<S, 3, 3>& T,
                              const TensorView_<S, 2>& X) -> Tensor_<S, 2>
  {
    auto TX = Tensor_<S, 2>{X.sizes()};
    auto TX_ = TX.colmajor_view().matrix();

    const auto X_matrix = X.colmajor_view().matrix();

    if (X.size(1) == 2)
      TX_ = (T * X_matrix.colwise().homogeneous()).colwise().hnormalized();
    else if (X.size(1) == 3)
    {
      TX_ = T * X_matrix;
      TX_.array().rowwise() /= TX_.array().row(2);
    }
    else
      throw std::runtime_error{"To apply the transform the input data "
                               "dimension must be 2 or 3!"};

    return TX;
  }

  template <typename Model>
  struct Normalizer
  {
  };

  template <typename S>
  struct Normalizer<Projective::Line2<S>>
  {
    inline Normalizer(const TensorView_<S, 2>& X)
      : T{compute_normalizer(X)}
    {
    }

    inline auto normalize(const PointList<S, 2>& X) const
    {
      auto Xn = PointList<S, 2>{};
      Xn._data = apply_transform(T, X._data);
      return Xn;
    }

    inline auto denormalize(Projective::Line2<S>& l) const -> void
    {
      l = T.transpose() * l;
    }

    Eigen::Matrix3<S> T;
  };

  //! @brief Dummy specialization for the orthogonal vanishing point problem...
  template <typename T>
  struct Normalizer<Eigen::Matrix3<T>>
  {
    inline auto normalize(const PointList<T, 2>& X) const -> PointList<T, 2>
    {
      return X;
    }

    //! @brief Dummy implementation.
    inline auto denormalize(const Eigen::Matrix3<T>&) const -> void
    {
    }
  };

}  // namespace DO::Sara
