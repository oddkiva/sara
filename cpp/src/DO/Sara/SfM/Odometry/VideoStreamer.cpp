#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/SfM/Odometry/VideoStreamer.hpp>


using namespace DO::Sara;


VideoStreamer::VideoStreamer(const std::filesystem::path& video_path)
{
  open(video_path);
}

auto VideoStreamer::open(const std::filesystem::path& video_path) -> void
{
  _frame_index = -1;
  _video_stream.close();
  _video_stream.open(video_path);
  _rgb8.swap(_video_stream.frame());
  _gray32f.resize(_video_stream.sizes());
}

auto VideoStreamer::read() -> bool
{
  const auto read_new_frame = _video_stream.read();
  from_rgb8_to_gray32f(_rgb8, _gray32f);
  if (read_new_frame)
    ++_frame_index;
  return read_new_frame;
}
