#pragma once

#include <array>
#include <memory>
#include <string>


class JpegImageReader
{
  struct Impl;

public:
  JpegImageReader() = default;

  explicit JpegImageReader(const std::string& filepath);

  auto imageSizes() const -> std::array<int, 3>;

  auto read(unsigned char* data) -> void;

private:
  std::string _fp;
  std::shared_ptr<Impl> _impl;
};
