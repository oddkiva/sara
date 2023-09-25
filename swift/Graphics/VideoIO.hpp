#pragma once

#include <swift/bridging>

#include <array>
#include <memory>
#include <string>


class VideoStreamImpl
{
  struct Impl;

public:
  VideoStreamImpl() = default;

  VideoStreamImpl(const VideoStreamImpl&) = default;

  explicit VideoStreamImpl(const std::string& filepath);

  unsigned char* framePointer() const SWIFT_RETURNS_INDEPENDENT_VALUE;

  auto frameWidth() const -> int;

  auto frameHeight() const -> int;

  auto read() -> bool;

private:
  std::string _fp;
  std::shared_ptr<Impl> _impl;
};
