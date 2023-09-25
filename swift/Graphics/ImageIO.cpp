#include "ImageIO.hpp"

#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>


struct JpegImageReader::Impl
{
  explicit Impl(const std::string& filepath)
    : _internal{filepath.c_str()}
  {
  }

  DO::Sara::JpegFileReader _internal;
};

JpegImageReader::JpegImageReader(const JpegImageReader& other)
  : _fp{other._fp}
  , _impl{new Impl{other._fp}}
{
}

JpegImageReader::JpegImageReader(const std::string& filepath)
  : _fp{filepath}
  , _impl{new Impl{filepath}}
{
}

auto JpegImageReader::imageSizes() const -> std::array<int, 3>
{
  const auto [w, h, c] = _impl->_internal.image_sizes();
  return {w, h, c};
}

auto JpegImageReader::read(unsigned char* data) -> void
{
  _impl->_internal.read(data);
}
