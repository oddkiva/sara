#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  const string video_filepath = src_path("orion_1.mpg");

  VideoStream video_stream(video_filepath);
  auto video_frame = Image<Rgb8>{video_stream.sizes()};

  while (true)
  {
    video_stream >> video_frame;
    if (!active_window())
      create_window(video_frame.sizes());

    if (!video_frame.data())
      break;
    display(video_frame);
  }

  return 0;
}
