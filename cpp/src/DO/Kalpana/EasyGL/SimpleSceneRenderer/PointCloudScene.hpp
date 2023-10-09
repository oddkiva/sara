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
    auto set_viewport_box(const AxisAlignedBoundingBox<int>& box) -> void
    {
      BasicScene::_viewport = Viewport{box};
      _projection = _viewport.perspective_projection(80.f, 0.5f, 5000.f);
      _model_view = _point_cloud_camera.view_matrix();
      // CAVEAT: re-express the point cloud in OpenGL axis convention.
      gl_Rt_cam.setIdentity();
      gl_Rt_cam.matrix().topLeftCorner<3, 3>() = gl_R_cam;
    }

    auto init() -> void
    {
      _checkerboard_renderer.initialize();
      _point_cloud.initialize();
      _point_cloud_renderer.initialize();
    }

    auto deinit() -> void
    {
      _checkerboard_renderer.destroy();
      _point_cloud.destroy();
      _point_cloud_renderer.destroy();
    }

    auto render() -> void
    {
      glViewport(_viewport.top_left().x(), _viewport.top_left().y(),
                 _viewport.width(), _viewport.height());

      // Render the checkerboard.
      _checkerboard_renderer.render(Eigen::Matrix4f::Identity(), _model_view,
                                    _projection);

      // Render the point cloud.
      _point_cloud_renderer.render(_point_cloud, _point_size,
                                   gl_Rt_cam.matrix(),  //
                                   _model_view, _projection);
    }

    auto update_point_cloud(const TensorView_<float, 2>& colored_point_cloud)
        -> void
    {
      _point_cloud.upload_host_data_to_gl(colored_point_cloud);
    }

    //! @brief Checkerboard renderer
    CheckerboardRenderer _checkerboard_renderer;

    //! Point cloud rendering
    //!
    //! @brief Point cloud GPU data.
    ColoredPointCloud _point_cloud;
    //! @brief Point cloud GPU renderer.
    ColoredPointCloudRenderer _point_cloud_renderer;
    //! @brief Point cloud rendering options.
    Camera _point_cloud_camera;
    float _point_size = 3.f;

    const Eigen::Matrix3f gl_R_cam =
        axis_permutation_matrix(AxisConvention::OpenGL).cast<float>();
    Eigen::Transform<float, 3, Eigen::Projective> gl_Rt_cam;
  };


}  // namespace DO::Kalpana::GL
