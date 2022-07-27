#include "ImageOrVideoReader.hpp"

#include <boost/filesystem.hpp>


namespace DO::Sara {

  ImageOrVideoReader::ImageOrVideoReader(const std::string& p)
  {
    open(p);
    if (_is_image)
      read();
  }

  auto ImageOrVideoReader::open(const std::string& path) -> void
  {
    namespace fs = boost::filesystem;
    if (fs::path{path}.extension().string() == ".png")
    {
      _path = path;
      _is_image = true;
    }
    else
      VideoStream::open(path);
  }

  auto ImageOrVideoReader::read() -> bool
  {
    if (_is_image && _frame.empty())
    {
      _frame = imread<Rgb8>(_path);
      return true;
    }
    else if (!_is_image)
      return VideoStream::read();

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
    if (_is_image)
      return {_frame.data(), _frame.sizes()};
    else
      return VideoStream::frame();
  }

}  // namespace DO::Sara
