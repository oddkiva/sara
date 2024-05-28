#pragma once

#include <DO/Sara/Core/Math/AxisConvention.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL/Objects/Camera.hpp>
#include <DO/Kalpana/EasyGL/Objects/ColoredPointCloud.hpp>
#include <DO/Kalpana/EasyGL/Objects/Scene.hpp>
#include <DO/Kalpana/EasyGL/Renderer/CheckerboardRenderer.hpp>
#include <DO/Kalpana/EasyGL/Renderer/ColoredPointCloudRenderer.hpp>


namespace DO::Kalpana::GL {

  struct PointCloudScene : BasicScene
  {
    PointCloudScene()
    {
      // CAVEAT: re-express the point cloud in OpenGL axis convention.
      const Eigen::Matrix3f P3 =
          axis_permutation_matrix(Sara::AxisConvention::OpenGL).cast<float>();
      _P.setIdentity();
      _P.matrix().topLeftCorner<3, 3>() = P3;
    }

    auto set_viewport_box(const Sara::AxisAlignedBoundingBox<int, 2>& box)
        -> void
    {
      _viewport = Viewport{box};
    }

    auto init() -> void
    {
      _checkerboard_renderer.initialize();
      _point_vbo.initialize();
      _point_renderer.initialize();
    }

    auto deinit() -> void
    {
      _checkerboard_renderer.destroy();
      _point_vbo.destroy();
      _point_renderer.destroy();
    }

    auto render() -> void
    {
      glViewport(_viewport.top_left().x(), _viewport.top_left().y(),
                 _viewport.width(), _viewport.height());

      // Render the checkerboard.
      _checkerboard_renderer.render(Eigen::Matrix4f::Identity(), _model_view,
                                    _projection);

      // Render the point cloud.
      _point_renderer.render(_point_vbo, _point_size, _P.matrix(), _model_view,
                             _projection);
    }

    auto
    update_point_cloud(const Sara::TensorView_<float, 2>& colored_point_cloud)
        -> void
    {
      _point_vbo.upload_host_data_to_gl(colored_point_cloud);
    }

    //! @brief Checkerboard renderer
    CheckerboardRenderer _checkerboard_renderer;

    //! Point cloud rendering
    //!
    //! @brief Point cloud GPU data.
    ColoredPointCloud _point_vbo;
    //! @brief Point cloud GPU renderer.
    ColoredPointCloudRenderer _point_renderer;

    //! @brief Point cloud rendering options.
    Camera _camera;
    float _point_size = 3.f;
    //! @brief The conversion matrix from the computer vision axis convention to
    //! OpenGL axis convention.
    Eigen::Transform<float, 3, Eigen::Projective> _P;
  };


}  // namespace DO::Kalpana::GL
