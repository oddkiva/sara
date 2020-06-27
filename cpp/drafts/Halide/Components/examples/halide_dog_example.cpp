#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/DoGExtremum.hpp>
#include <drafts/Halide/Components/GaussianConvolution2D.hpp>
#include <drafts/Halide/Components/LocalExtremum.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace hal = DO::Shakti::HalideBackend;
namespace sara = DO::Sara;


using namespace std;


auto x = Halide::Var{"x"};
auto y = Halide::Var{"y"};
auto c = Halide::Var{"c"};

auto xo = Halide::Var{"xi"};
auto yo = Halide::Var{"yi"};
auto co = Halide::Var{"ci"};

auto xi = Halide::Var{"xi"};
auto yi = Halide::Var{"yi"};
auto ci = Halide::Var{"ci"};

const auto tile_x = 16;
const auto tile_y = 16;


auto rescale_to_rgb(const Halide::Func& f, int32_t w, int32_t h)
{
  auto r = Halide::RDom(0, w, 0, h);
  auto f_min = Halide::Func{f.name() + "_max"};
  auto f_max = Halide::Func{f.name() + "_min"};
  f_max() = Halide::maximum(f(r.x, r.y));
  f_min() = Halide::minimum(f(r.x, r.y));
  f_max.compute_root();
  f_min.compute_root();

  auto f_rescaled = Halide::Func{f.name() + "_rescaled"};
  f_rescaled(x, y, c) = cast<std::uint8_t>(  //
      (f(x, y) - f_min()) /                  //
      (f_max() - f_min()) * 255              //
  );

  return f_rescaled;
}

auto schedule(Halide::Func& f, bool gpu)
{
  if (gpu)
    f.gpu_tile(x, y, xi, yi, tile_x, tile_y, Halide::TailStrategy::GuardWithIf);
  else
    f.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
}


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath =
      "/Users/david/Desktop/Datasets/humanising-autonomy/turn_bikes.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  sara::VideoStream video_stream(video_filepath);

  // Input and output images.
  auto frame = video_stream.frame();
  auto dog_frame = sara::Image<sara::Rgb8>{frame.sizes()};

  auto input = hal::as_interleaved_buffer(frame);
  auto output = hal::as_interleaved_buffer(dog_frame);

  constexpr auto use_gpu = true;
  const auto jit_target =
      use_gpu ? hal::get_gpu_target() : hal::get_host_target();

  auto input_ext = Halide::BoundaryConditions::repeat_edge(input);

  // auto downsized = Halide::Func{"downsized"};
  // downsized(x, y, c) = (input_ext(2 * x + 0, 2 * y + 0, c) +   //
  //                       input_ext(2 * x + 0, 2 * y + 1, c) +   //
  //                       input_ext(2 * x + 1, 2 * y + 0, c) +   //
  //                       input_ext(2 * x + 1, 2 * y + 1, c)) /  //
  //                      4.f;                                      //

  auto gray = Halide::Func{"gray"};
  {
    auto r = input_ext(x, y, 0);
    auto g = input_ext(x, y, 1);
    auto b = input_ext(x, y, 2);
    gray(x, y) = (0.2125f * r + 0.7154f * g + 0.0721f * b) / 255.f;
    gray.compute_root();
  }

  auto num_scales = 3 + 3;
  auto delta_s = static_cast<float>(std::pow(2, 1. / 3.));
  auto sigmas = std::vector<float>(num_scales);
  for (auto s = 0; s < num_scales; ++s)
    sigmas[s] = std::pow(delta_s, s + 1);

  auto gauss_pyr = std::vector<hal::GaussianConvolution2D<>>{};
  for (auto s = 0; s < num_scales; ++s)
  {
    auto gauss = hal::GaussianConvolution2D<>{};
    gauss.output = Halide::Func{"gauss_diff"};
    gauss.generate_2d(gray, sigmas[s], 4, frame.width(), frame.height());
    gauss.schedule_2d(jit_target);
    gauss.output.compute_root();

    gauss_pyr.push_back(gauss);
  }

  auto dog_pyr = std::vector<Halide::Func>(num_scales - 1);
  for (auto s = 0; s < num_scales - 1; ++s)
  {
    dog_pyr[s](x, y) =
        gauss_pyr[s + 1].output(x, y) - gauss_pyr[s].output(x, y);
    dog_pyr[s].compute_root();
    schedule(dog_pyr[s], use_gpu);
  }

  auto dog_local_max = Halide::Func{"dog_local_max"};
  {
    auto is_local_max =
        hal::local_scale_space_max(dog_pyr[0], dog_pyr[1], dog_pyr[2], x, y);
    auto edge_ratio = Expr{};
    edge_ratio = 4.f;
    auto is_not_on_edge = !hal::on_edge(dog_pyr[1], edge_ratio, x, y);
    dog_local_max(x, y) = Halide::cast<float>(is_local_max && is_not_on_edge); //* dog_pyr[1](x, y);
  }
  schedule(dog_local_max, use_gpu);
  dog_local_max.compute_root();

  auto output_as_rgb = rescale_to_rgb(dog_local_max, frame.width(), frame.height());
  output_as_rgb.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);
  schedule(output_as_rgb, use_gpu);
  output_as_rgb.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);
  output_as_rgb.compile_jit(jit_target);

  sara::create_window(video_stream.sizes());
  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    sara::tic();
    {
      input.set_host_dirty();
      output_as_rgb.realize(output);
      output.copy_to_host();
    }
    sara::toc("DoG");

    sara::display(dog_frame);
  }

  return 0;
}
