#include "CGraphics.hpp"

#include <DO/Sara/Graphics/DerivedQObjects/GraphicsContext.hpp>
#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <QApplication>


namespace sara = DO::Sara;


auto JpegImageReader_init(const char* filepath) -> void*
{
  auto reader = new sara::JpegFileReader(filepath);
  return reinterpret_cast<void*>(reader);
}

auto JpegImageReader_deinit(void* reader) -> void
{
  delete reinterpret_cast<sara::JpegFileReader*>(reader);
}

auto JpegImageReader_imageSizes(void* reader, int* w, int* h, int* c) -> void
{
  auto r = reinterpret_cast<sara::JpegFileReader*>(reader);
  std::tie(*w, *h, *c) = r->image_sizes();
}

auto JpegImageReader_readImageData(void* reader, unsigned char* dataPtr) -> void
{
  auto r = reinterpret_cast<sara::JpegFileReader*>(reader);
  r->read(dataPtr);
}


auto VideoStream_init(const char* filepath) -> void*
{
  auto reader = new sara::VideoStream{filepath};
  return reinterpret_cast<void*>(reader);
}

auto VideoStream_deinit(void* stream) -> void
{
  delete reinterpret_cast<sara::VideoStream*>(stream);
}

auto VideoStream_getFramePtr(void* stream) -> unsigned char *
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  auto frame = vstream->frame();
  auto framePtr = reinterpret_cast<unsigned char*>(frame.data());
  return framePtr;
}

auto VideoStream_getFrameWidth(void* stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return vstream->width();
}

auto VideoStream_getFrameHeight(void* stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return vstream->height();
}

auto VideoStream_readFrame(void *stream) -> int
{
  auto vstream = reinterpret_cast<sara::VideoStream *>(stream);
  return static_cast<int>(vstream->read());
}
