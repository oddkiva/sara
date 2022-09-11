#include "ImageOrVideoReader.hpp"

#include <boost/filesystem.hpp>


namespace DO::Sara {

  ImageOrVideoReader::ImageOrVideoReader(const std::string& p)
  {
    open(p);
    if (_is_png_image)
      read();
  }

  auto ImageOrVideoReader::open(const std::string& path) -> void
  {
    namespace fs = boost::filesystem;
    if (fs::path{path}.extension().string() == ".png")
    {
      _path = path;
      _is_png_image = true;
    }
    else
    {
      _is_png_image = false;
      // FFmpeg can also decode jpeg images =).
      _video_stream.open(path);
    }
  }

  auto ImageOrVideoReader::read() -> bool
  {
    if (_is_png_image && _frame.empty())
    {
      _frame = imread<Rgb8>(_path);
      return true;
    }
    else if (!_is_png_image)
      return _video_stream.read();

    // Horrible hack, well...
    if (!_read_once)
    {
      _read_once = true;
      return true;
    }
    else
      return false;
  }

  auto ImageOrVideoReader::frame() -> ImageView<Rgb8>
  {
    if (_is_png_image)
      return {_frame.data(), _frame.sizes()};
    else
      return _video_stream.frame();
  }

}  // namespace DO::Sara
