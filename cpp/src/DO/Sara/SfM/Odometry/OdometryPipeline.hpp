#pragma once

#include <DO/Sara/Graphics/ImageDraw.hpp>
#include <DO/Sara/SfM/Odometry/FeatureTracker.hpp>
#include <DO/Sara/SfM/Odometry/ImageDistortionCorrector.hpp>
#include <DO/Sara/SfM/Odometry/RelativePoseEstimator.hpp>
#include <DO/Sara/SfM/Odometry/Triangulator.hpp>
#include <DO/Sara/SfM/Odometry/VideoStreamer.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>


namespace DO::Sara {

  struct OdometryPipeline
  {
    auto set_config(const std::filesystem::path& video_path,
                    const v2::BrownConradyDistortionModel<double>& camera)
        -> void
    {
      // Build the dependency graph.
      _video_streamer.open(video_path);
      _camera = camera;

      _distortion_corrector = std::make_unique<ImageDistortionCorrector>(
          _video_streamer.frame_rgb8(),     //
          _video_streamer.frame_gray32f(),  //
          _camera                           //
      );

      _feature_tracker = std::make_unique<FeatureTracker>(
          _distortion_corrector->frame_gray32f());

      _relative_pose_estimator = std::make_unique<RelativePoseEstimator>(
          _feature_tracker->keys, _feature_tracker->matches, _camera);

      _triangulator = std::make_unique<Triangulator>(
          _relative_pose_estimator->_geometry.C1,
          _relative_pose_estimator->_geometry.C2,  //
          _relative_pose_estimator->_K, _relative_pose_estimator->_K_inv,
          _relative_pose_estimator->_X, _relative_pose_estimator->_inliers);
    }

    auto read() -> bool
    {
      return _video_streamer.read();
    }

    auto process() -> void
    {
      if (_video_streamer.skip())
        return;

      _distortion_corrector->undistort();

      // N.B.: detect the features on the **undistorted** image.
      _feature_tracker->detect_features();
      _feature_tracker->match_features();

      const auto success = _relative_pose_estimator->estimate_relative_pose();
      if (!success)
        return;

      _triangulator->triangulate();
      _triangulator->extract_colors(_distortion_corrector->frame_rgb8(0),
                                    _distortion_corrector->frame_rgb8(1));
      _triangulator->update_colored_point_cloud();
    }

    auto make_display_frame() const -> Image<Rgb8>
    {
      Image<Rgb8> display = _distortion_corrector->frame_rgb8();
      const auto& matches = _feature_tracker->matches;
      const auto& inliers = _relative_pose_estimator->_inliers;
      const auto num_matches = static_cast<int>(matches.size());
#pragma omp parallel for
      for (auto m = 0; m < num_matches; ++m)
      {
        if (!inliers(m))
          continue;
        const auto& match = matches[m];
        draw(display, match.x(), Blue8);
        draw(display, match.y(), Cyan8);
        draw_arrow(display, match.x_pos(), match.y_pos(), Yellow8);
      }

      return display;
    }

    VideoStreamer _video_streamer;
    v2::BrownConradyDistortionModel<double> _camera;

    std::unique_ptr<ImageDistortionCorrector> _distortion_corrector;
    std::unique_ptr<FeatureTracker> _feature_tracker;
    std::unique_ptr<RelativePoseEstimator> _relative_pose_estimator;
    std::unique_ptr<Triangulator> _triangulator;
  };

}  // namespace DO::Sara
