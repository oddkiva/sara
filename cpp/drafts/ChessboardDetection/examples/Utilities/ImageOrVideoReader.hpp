#pragma once

#include <boost/filesystem.hpp>

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace DO::Sara {

  struct ImageOrVideoReader
  {
    inline ImageOrVideoReader() = default;

    ImageOrVideoReader(const std::string& path);

    auto open(const std::string& path) -> void;

    auto read() -> bool;

    auto frame() -> ImageView<Rgb8>;

    bool _is_png_image = false;

    VideoStream _video_stream;

    std::string _path;
    Image<Rgb8> _frame;
    bool _read_once = false;
  };

}  // namespace DO::Sara
