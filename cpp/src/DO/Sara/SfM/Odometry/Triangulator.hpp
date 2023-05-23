#pragma once

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>


namespace DO::Sara {

  struct Triangulator
  {
    Triangulator(const PinholeCameraDecomposition& C1,
                 const PinholeCameraDecomposition& C2,  //
                 const Eigen::Matrix3d& K,              //
                 const Eigen::Matrix3d& K_inv,
                 const PointCorrespondenceList<double>& X,
                 const TensorView_<bool, 1>& inliers)

      : _camera_1{C1}
      , _camera_2{C2}
      , _K{K}
      , _K_inv{K_inv}
      , _X{X}
      , _inliers{inliers}
    {
    }

    //! @brief Input data.
    const PinholeCameraDecomposition& _camera_1;
    const PinholeCameraDecomposition& _camera_2;
    const Eigen::Matrix3d& _K;
    const Eigen::Matrix3d& _K_inv;
    const PointCorrespondenceList<double>& _X;
    const TensorView_<bool, 1>& _inliers;

    //! @brief Output data.
    TwoViewGeometry _geometry;
    Tensor_<double, 2> _colors;
    Tensor_<float, 2> _colored_point_cloud;

    auto triangulate() -> void
    {
      print_stage("Retriangulating the inliers...");
      auto& points = _geometry.X;
      auto& s1 = _geometry.scales1;
      auto& s2 = _geometry.scales2;
      points.resize(4, _inliers.flat_array().count());
      s1.resize(_inliers.flat_array().count());
      s2.resize(_inliers.flat_array().count());
      auto cheiral_inlier_count = 0;

      auto& j = cheiral_inlier_count;
      for (auto i = 0; i < _inliers.size(0); ++i)
      {
        if (!_inliers(i))
          continue;

        const Eigen::Vector3d u1 = _K_inv * _X[i][0].vector();
        const Eigen::Vector3d u2 = _K_inv * _X[i][1].vector();
        const auto [Xj, s1j, s2j] = triangulate_single_point_linear_eigen(
            _camera_1.matrix(), _camera_2.matrix(), u1, u2);
        const auto cheiral = s1j > 0 && s2j > 0;
        if (!cheiral)
          continue;

        // Also we want z in [0, 200] meters max...
        // We want to avoid 3D point corresponding to the sky...
        Eigen::Vector4d Xjh = Xj.hnormalized().homogeneous();
#if 0
        const auto reasonable = 0 < Xjh.z() && Xjh.z() < 200;
        if (!reasonable)
          continue;
#endif

        points.col(j) = Xjh;
        s1(j) = s1j;
        s2(j) = s2j;
        ++j;
      }

      SARA_CHECK(cheiral_inlier_count);
      points = points.leftCols(cheiral_inlier_count);
      s1 = s1.head(cheiral_inlier_count);
      s2 = s2.head(cheiral_inlier_count);
    }

    auto extract_colors(const ImageView<Rgb8>& frame_prev,
                        const ImageView<Rgb8>& frame_curr) -> void
    {
      print_stage("Extracting the colors...");

      // TODO: fix this because this is very wrong from a design point of view.
      _geometry.C1.K = _K;
      _geometry.C2.K = _K;

      _colors = Sara::extract_colors(frame_prev,
                                     frame_curr,  //
                                     _geometry);
    }

    auto update_colored_point_cloud() -> void
    {
      print_stage("Updating the colored point cloud...");
      _colored_point_cloud.resize(_colors.size(0), 6);
      SARA_CHECK(_colored_point_cloud.sizes().transpose());
      SARA_CHECK(_colors.sizes().transpose());
      SARA_CHECK(_geometry.X.rows());
      SARA_CHECK(_geometry.X.cols());
      const auto& X = _geometry.X.colwise().hnormalized().transpose();
      _colored_point_cloud.matrix().leftCols(3) = X.cast<float>();
      _colored_point_cloud.matrix().rightCols(3) =
          _colors.matrix().cast<float>();
    }
  };

}  // namespace DO::Sara
