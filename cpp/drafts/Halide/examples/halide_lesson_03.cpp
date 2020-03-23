#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "../shakti_halide_utilities.hpp"


namespace halide = DO::Shakti::HalideBackend;


using namespace std;
using namespace DO::Sara;


auto halide_pipeline() -> void
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      "/Users/david/GitLab/DO-CV/sara/cpp/examples/Sara/VideoIO/orion_1.mpg"s;
#else
  // const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  VideoStream video_stream(video_filepath);

  // Input and output images.
  auto input_image = video_stream.frame();
  auto output_image = Image<Rgb8>{video_stream.sizes()};

  // Halide input and output buffers.
  auto input_buffer = halide::as_interleaved_rgb_buffer(input_image);
  auto output_buffer = halide::as_interleaved_rgb_buffer(output_image);

  // Blur pipeline.
  // const auto pipeline = halide::Blur3x3Vis{input_buffer};
  // const auto pipeline = halide::LaplacianVis{input_buffer};
  const auto to_gray = halide::RgbToGray32f{input_buffer};

  create_window(video_stream.sizes());
  while (true)
  {
    tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    toc("Video Decoding");

    tic();
    //pipeline(output_buffer);
    toc("Halide");

    display(output_image);
  }
}


GRAPHICS_MAIN()
{
  halide_pipeline();
  return 0;
}
