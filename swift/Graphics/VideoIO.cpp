#include "VideoIO.hpp"

#include <DO/Sara/VideoIO/VideoStream.hpp>


struct VideoStreamImpl::Impl
{
  Impl() = default;

  Impl(const Impl&) = delete;

  explicit Impl(const std::string& filepath)
    : _internal{filepath}
  {
  }

  DO::Sara::VideoStream _internal;
};

VideoStreamImpl::VideoStreamImpl(const std::string& filepath)
  : _fp{filepath}
  , _impl{new Impl{filepath}}
{
  std::cout << _fp << std::endl;
}

auto VideoStreamImpl::framePointer() const -> unsigned char*
{
  auto frame = _impl->_internal.frame();
  auto framePtr = reinterpret_cast<unsigned char*>(frame.data());
  return framePtr;
}

auto VideoStreamImpl::frameWidth() const -> int
{
  return _impl->_internal.width();
}

auto VideoStreamImpl::frameHeight() const -> int
{
  return _impl->_internal.height();
}

auto VideoStreamImpl::read() -> bool
{
  return _impl->_internal.read();
}
