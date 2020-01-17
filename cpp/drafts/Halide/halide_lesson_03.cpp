#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include "Halide.h"


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  const string video_filepath = "/home/david/Desktop/test.mp4";

  VideoStream video_stream(video_filepath);

  // Input.
  auto video_frame = Image<Rgb8>{video_stream.sizes()};

  // Output.
  auto video_frame_processed = video_frame;

  // Timer.
  auto timer = Timer{};

  // Image processing pipeline.
  auto input =
      Halide::Buffer<uint8_t>{reinterpret_cast<uint8_t*>(video_frame.data()),
                              {video_frame.width(), video_frame.height(), 3}};
  auto output = Halide::Buffer<uint8_t>{
      reinterpret_cast<uint8_t*>(video_frame_processed.data()),
      {video_frame_processed.width(), video_frame_processed.height(), 3}};

  auto x = Halide::Var{};
  auto y = Halide::Var{};
  auto c = Halide::Var{};

  auto cast = Halide::Func{};
  cast(x, y, c) = input(x, y, c) / 255.f;

  auto padded = Halide::Func{};
  padded(x, y, c) =
      cast(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1), c);

  auto filter = Halide::Func{};
  //filter(x, y, c) =
  //    padded(x, y, c) - (padded(x + 1, y + 0, c) + padded(x - 1, y + 0, c) +
  //                       padded(x + 0, y + 1, c) + padded(x + 0, y - 1, c)) /
  //                          4.f;

  // filter(x, y, c) =
  //     padded(x, y, c) - (padded(x + 1, y + 0, c) + padded(x - 1, y + 0, c)) / 2.f;

  filter(x, y, c) = (padded(x, y + 1, c) - padded(x, y - 1, c)) / 2.f;

  auto filter_rescaled = Halide::Func{};
  filter_rescaled(x, y, c) = Halide::cast<uint8_t>((filter(x, y, c) + 0.5f) * 255.f);

  // Scheduling.
  filter_rescaled.reorder(c, x, y).bound(c, 0, 3).unroll(c);

  while (true)
  {
    video_stream.read(video_frame);
    if (!active_window())
      create_window(video_frame.sizes());

    if (!video_frame.data())
      break;

    {
      timer.restart();
      filter_rescaled.realize(output);
      const auto elapsed = timer.elapsed_ms();
      std::cout << "Computation time = " << elapsed << " ms" << std::endl;
    }

    display(video_frame_processed);
  }

  return 0;
}
