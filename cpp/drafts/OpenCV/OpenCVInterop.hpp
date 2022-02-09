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

    inline auto operator()(int i, int j) const -> const Eigen::Vector2f
    {
      const auto index = i * _corner_count_per_axis.width + j;
      const auto& p = _corners[index];
      return {p.x, p.y};
    }

    inline auto point_3d(int i, int j) const -> Eigen::Vector2f
    {
      return {_square_size_in_meters * i, _square_size_in_meters * j};
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
    for (auto i = 0; i < h; ++i)
    {
      const auto ratio = static_cast<float>(i) / h;
      const auto color = rainbow_color(ratio);

      for (auto j = 0; j < w; ++j)
      {
        c0 = c1;
        c1 = chessboard(i, j);
        draw_circle(image, c1, 3.f, color, 2);

        if (i == 0 && j == 0)
          continue;
        draw_arrow(image, c0, c1, color, 2);
      }
    }
  }


}  // namespace DO::Sara::OpenCV
