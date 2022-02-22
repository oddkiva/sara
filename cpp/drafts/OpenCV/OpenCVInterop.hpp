#pragma once

#include <opencv2/opencv.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>


namespace DO::Sara::OpenCV {

  inline auto to_cv_mat(const ImageView<Rgb8>& imview) -> cv::Mat
  {
    return {imview.height(), imview.width(), CV_8UC3,
            reinterpret_cast<void*>(const_cast<Rgb8*>(imview.data()))};
  }

  class Chessboard
  {
  public:
    inline Chessboard() = default;

    inline explicit Chessboard(const Eigen::Vector2i& corner_count_per_axis,
                               float square_size_in_meters)
      : _corner_count_per_axis{corner_count_per_axis(0),
                               corner_count_per_axis(1)}
      , _square_size_in_meters{square_size_in_meters}
    {
    }

    inline auto width() const -> int
    {
      return _corner_count_per_axis.width;
    }

    inline auto height() const -> int
    {
      return _corner_count_per_axis.height;
    }

    inline auto corner_count() const -> int
    {
      return width() * height();
    }

    inline auto detect(const ImageView<Rgb8>& image) -> bool
    {
      static auto cv_frame_bgr = cv::Mat{};

      auto cv_frame_rgb = to_cv_mat(image);

      cv::cvtColor(cv_frame_rgb, cv_frame_bgr, cv::COLOR_RGB2BGR);
      const auto corners_found = cv::findChessboardCorners(  //
          cv_frame_bgr,                                      //
          _corner_count_per_axis,                            //
          _corners);

      return corners_found;
    }

    inline auto image_point(int x, int y) const -> Eigen::Vector2f
    {
      const auto index = y * _corner_count_per_axis.width + x;
      const auto& p = _corners[index];
      return {p.x, p.y};
    }

    inline auto scene_point(int x, int y) const -> Eigen::Vector2f
    {
      return {_square_size_in_meters * x, _square_size_in_meters * y};
    }

    inline auto corners_count_per_axis() const -> const Eigen::Vector2i
    {
      return {_corner_count_per_axis.width, _corner_count_per_axis.height};
    }

  private:
    cv::Size _corner_count_per_axis;
    float _square_size_in_meters;
    std::vector<cv::Point2f> _corners;
  };

  // From:
  // https://stackoverflow.com/questions/40629345/fill-array-dynamicly-with-gradient-color-c
  inline auto rainbow_color(float ratio) -> Rgb8
  {
    static constexpr auto quantization = 256;
    static constexpr auto region_count = 6;

    // We want to normalize the ratio so that it fits in to 6 regions where each
    // region is 256 units long.
    const auto normalized =
        static_cast<int>(ratio * quantization * region_count);

    // Find the region for this position.
    const auto region = normalized / quantization;

    // Find the distance to the start of the closest region.
    const auto x = normalized % quantization;

    auto r = std::uint8_t{};
    auto g = std::uint8_t{};
    auto b = std::uint8_t{};

    switch (region)
    {
      // clang-format off
    case 0: r = 255; g = 0;   b = 0;   g += x; break;
    case 1: r = 255; g = 255; b = 0;   r -= x; break;
    case 2: r = 0;   g = 255; b = 0;   b += x; break;
    case 3: r = 0;   g = 255; b = 255; g -= x; break;
    case 4: r = 0;   g = 0;   b = 255; r += x; break;
    case 5: r = 255; g = 0;   b = 255; b -= x; break;
      // clang-format on
    }

    return {r, g, b};
  }

  inline auto draw_chessboard(ImageView<Rgb8>& image,
                              const OpenCV::Chessboard& chessboard) -> void
  {
    const auto& pattern_size = chessboard.corners_count_per_axis();
    const auto& w = pattern_size.x();
    const auto& h = pattern_size.y();

    auto c0 = Eigen::Vector2f{};
    auto c1 = Eigen::Vector2f{};
    for (auto y = 0; y < h; ++y)
    {
      const auto ratio = static_cast<float>(y) / h;
      const auto color = rainbow_color(ratio);

      for (auto x = 0; x < w; ++x)
      {
        c0 = c1;
        c1 = chessboard.image_point(x, y);
        draw_circle(image, c1, 3.f, color, 2);

        if (x == 0 && y == 0)
          continue;
        draw_arrow(image, c0, c1, color, 2);
      }
    }
  }

  inline auto inspect(ImageView<Rgb8>& image,          //
                      const OpenCV::Chessboard& chessboard,  //
                      const Eigen::Matrix3d& K,                    //
                      const Eigen::Matrix3d& R,                    //
                      const Eigen::Vector3d& t, bool pause = false) -> void
  {
    auto Hr = Eigen::Matrix3f{};
    Hr.col(0) = R.col(0).cast<float>();
    Hr.col(1) = R.col(1).cast<float>();
    Hr.col(2) = t.cast<float>();
    Hr = (K.cast<float>() * Hr).normalized();

    const auto a = chessboard.image_point(0, 0);
    const auto b = chessboard.image_point(0, 1);
    const auto c = chessboard.image_point(1, 0);

    const Eigen::Vector2f d = (K * (R * Eigen::Vector3d::UnitZ() + t))  //
                                  .hnormalized()
                                  .cast<float>();


    static const auto red = Rgb8{167, 0, 0};
    draw_arrow(image, a, b, red, 6);
    draw_circle(image, a, 5.f, red, 6);
    draw_circle(image, b, 5.f, red, 6);

    static const auto green = Rgb8{89, 216, 26};
    draw_arrow(image, a, c, green, 6);
    draw_circle(image, a, 5.f, green, 6);
    draw_circle(image, c, 5.f, green, 6);

    draw_arrow(image, a, d, Blue8, 6);
    draw_circle(image, a, 5.f, Blue8, 6);
    draw_circle(image, d, 5.f, Blue8, 6);

    for (auto y = 0; y < chessboard.height(); ++y)
    {
      for (auto x = 0; x < chessboard.width(); ++x)
      {
        const Eigen::Vector3f P = chessboard.scene_point(x, y).homogeneous();
        const Eigen::Vector2f p1 = chessboard.image_point(x, y);
        const Eigen::Vector2f p2 = (Hr * P).hnormalized();

        draw_circle(image, p1, 3.f, Cyan8, 3);
        draw_circle(image, p2, 3.f, Magenta8, 3);
        if (pause)
        {
          display(image);
          get_key();
        }
      }
    }
  }


}  // namespace DO::Sara::OpenCV
