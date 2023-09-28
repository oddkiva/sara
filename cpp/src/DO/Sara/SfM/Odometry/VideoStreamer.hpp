#pragma once

#include <DO/Sara/VideoIO/VideoStream.hpp>

#include <filesystem>


namespace DO::Sara {

  class VideoStreamer
  {
  public:
    VideoStreamer() = default;

    VideoStreamer(const std::filesystem::path& video_path);

    auto set_num_skips(const int num_skips) -> void
    {
      _num_skips = num_skips;
    }

    auto open(const std::filesystem::path& video_path) -> void;

    auto read() -> bool;

    auto frame_rgb8() const -> const ImageView<Rgb8>&
    {
      return _rgb8;
    }

    auto frame_gray32f() const -> const ImageView<float>&
    {
      return _gray32f;
    }

    auto sizes() const -> Eigen::Vector2i
    {
      return _video_stream.sizes();
    }

    auto width() const -> int
    {
      return _video_stream.width();
    }

    auto height() const -> int
    {
      return _video_stream.height();
    }

    auto skip() const -> bool
    {
      return _frame_index % (_num_skips + 1) != 0;
    }

  private:
    VideoStream _video_stream;
    ImageView<Rgb8> _rgb8;
    Image<float> _gray32f;

    int _num_skips = 2;
    int _frame_index = -1;
  };

}  // namespace DO::Sara
