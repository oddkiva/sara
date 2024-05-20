// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2024-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  //! @brief Two-point solver that solves the absolute translation given a known
  //!        absolute rotaton.
  //!        Also returns the scales for the two backprojected rays.
  //!
  //! Turns out to be very basic...
  template <typename T>
  struct AbsoluteTranslationSolver
  {
    static constexpr auto num_points = 2;
    static constexpr auto num_models = 1;

    using solution_type = Eigen::Vector<T, 5>;
    using translation_vector_type = Eigen::Vector3<T>;
    using scale_vector_type = Eigen::Vector2<T>;
    using model_type = std::pair<translation_vector_type, scale_vector_type>;

    using tensor_view_type = TensorView_<T, 2>;
    using data_point_type = std::array<TensorView_<T, 2>, 2>;

    //! @brief Inputs are **rotated** world scene points and the backprojected
    //! rays.
    auto operator()(const Eigen::Matrix<T, 3, 2>& Rx,
                    const Eigen::Matrix<T, 3, 2>& y) const -> model_type
    {
      const auto x0 = Rx.col(0);
      const auto x1 = Rx.col(1);
      const auto y0 = y.col(0);
      const auto y1 = y.col(1);

      static const auto I3 = Eigen::Matrix3<T>::Identity();
      static const auto O3 = Eigen::Vector3<T>::Zero();

      auto A = Eigen::Matrix<T, 6, 5>{};
      A.template topRows<3>() << -I3, y0, O3;
      A.template bottomRows<3>() << -I3, O3, y1;

      auto b = Eigen::Vector<T, 6>{};
      b << x0, x1;

      const solution_type x = A.colPivHouseholderQr().solve(b);
      const translation_vector_type t = x.head(3);
      const scale_vector_type scales = x.tail(2);

      return {t, scales};
    }
  };

  template <typename T>
  struct AbsolutePoseSolverUsingRotationKnowledge
  {
    //! @brief The translation solver.
    using translation_solver_type = AbsoluteTranslationSolver<T>;
    using tensor_view_type = TensorView_<T, 2>;
    using data_point_type = std::array<TensorView_<T, 2>, 2>;
    using model_type = Eigen::Matrix<T, 3, 4>;

    static constexpr auto num_points = translation_solver_type::num_points;
    static constexpr auto num_models = translation_solver_type::num_models;
    static_assert(num_models == 1);

    translation_solver_type tsolver;

    //! @brief The absolute rotation known.
    //!
    //! This can be set from the composition of success relative rotation.
    Eigen::Matrix3<T> R;

    inline auto operator()(const tensor_view_type& scene_points,
                           const tensor_view_type& rays) const
        -> std::array<model_type, num_models>
    {
      const auto sp_mat_ = scene_points.colmajor_view().matrix();

      Eigen::Matrix<T, 3, 2> sp_mat = sp_mat_.topRows(3);
      if (sp_mat_.cols() == 4)
        sp_mat.array().rowwise() /= sp_mat_.array().row(3);

      // Apply the global rotation to the scene points.
      sp_mat = R * sp_mat;

      const Eigen::Matrix<T, 3, 2> ray_mat = rays.colmajor_view().matrix();

      // Calculate the absolute translation.
      const auto [t, scales] = tsolver(sp_mat, ray_mat);

      // Return the absolute pose.
      auto pose = model_type{};
      pose << R, t;
      return {pose};
    }

    inline auto operator()(const data_point_type& X)
        -> std::array<model_type, num_models>
    {
      const auto& [scene_points, backprojected_rays] = X;
      return this->operator()(scene_points, backprojected_rays);
    }
  };
}  // namespace DO::Sara
