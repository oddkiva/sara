#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include "Halide.h"


GRAPHICS_MAIN()
{
  using namespace std;
  using namespace DO::Sara;

  // const string video_filepath = "/home/david/Desktop/test.mp4";
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4";

  VideoStream video_stream(video_filepath);

  // Input.
  auto in_video_frame = video_stream.frame();
  auto in_float = Image<float>{video_stream.sizes()};
  auto grad = Image<Vector2f>{video_stream.sizes()};
  auto grad_norm = Image<float>{video_stream.sizes()};
  auto grad_norm_rgb = Image<Rgb8>{video_stream.sizes()};

  auto in_video_frame_rgba = Image<Rgba8>{video_stream.sizes()};
  auto out_video_frame = Image<Rgba8>{video_stream.sizes()};

  // Timer.
  auto timer = Timer{};

  // Sara pipeline
  auto compute_gradient = Gradient{};

  // Halide pipeline.
  auto input = Halide::Buffer<uint8_t>{
      reinterpret_cast<uint8_t*>(in_video_frame_rgba.data()),
      {video_stream.width(), video_stream.height(), 4}};
  auto output = Halide::Buffer<uint8_t>{
      reinterpret_cast<uint8_t*>(out_video_frame.data()),
      {out_video_frame.width(), out_video_frame.height(), 4}};

  auto x = Halide::Var{};
  auto y = Halide::Var{};
  auto c = Halide::Var{};

  auto xo = Halide::Var{};
  auto yo = Halide::Var{};
  auto xi = Halide::Var{};
  auto yi = Halide::Var{};

  auto cast = Halide::Func{"cast"};
  cast(x, y, c) = input(x, y, c) / 255.f;

  auto padded = Halide::Func{"padded"};
  padded(x, y, c) = cast(clamp(x, 0, input.width() - 1),   //
                         clamp(y, 0, input.height() - 1),  //
                         c);

  auto filter = Halide::Func{"filter"};
  filter(x, y, c) =
      Halide::abs((padded(x + 1, y, c) - padded(x - 1, y, c)) / 2.f);

  auto filter_rescaled = Halide::Func{"filter_rescaled"};
  filter_rescaled(x, y, c) = Halide::cast<uint8_t>(filter(x, y, c) * 255.f);

  // // Schedule.
  // filter_rescaled.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);

  // // Run on CUDA.
  // auto target = Halide::get_host_target();
  // target.set_feature(Halide::Target::CUDA);
  // //target.set_feature(Halide::Target::Debug);

  // filter_rescaled.compile_jit(target);

  create_window(video_stream.sizes());

  auto elapsed = double{};

  while (true)
  {
    timer.restart();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Video decoding time = " << elapsed << " ms" << std::endl;

    timer.restart();
    {
      convert(in_video_frame, in_float);
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Color conversion time = " << elapsed << " ms" << std::endl;

    timer.restart();
    {
      compute_gradient(in_float, grad);
      std::transform(grad.begin(), grad.end(), grad_norm.begin(),
                     [](const Vector2f& g) { return g.norm(); });
      grad_norm = color_rescale(grad_norm);
      convert(grad_norm, grad_norm_rgb);
    }
    elapsed = timer.elapsed_ms();
    std::cout << "Sara computation time = " << elapsed << " ms" << std::endl;

    // timer.restart();
    //{
    //  filter_rescaled.realize(output);
    //  output.copy_to_host();
    //}
    // elapsed = timer.elapsed_ms();
    // std::cout << "Halide computation time = " << elapsed << " ms" <<
    // std::endl;

    display(grad_norm_rgb);
  }

  return 0;
}
