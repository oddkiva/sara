#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/GaussianConvolution2D.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace hal = DO::Shakti::HalideBackend;
namespace sara = DO::Sara;


using namespace std;


auto rescale_to_rgb(const Halide::Func& f, int32_t w, int32_t h)
{
  auto r = Halide::RDom(0, w, 0, h);
  auto f_min = Halide::Func{f.name() + "_max"};
  auto f_max = Halide::Func{f.name() + "_min"};
  f_max() = Halide::maximum(f(r.x, r.y));
  f_min() = Halide::minimum(f(r.x, r.y));
  f_max.compute_root();
  f_min.compute_root();

  auto x = Var{"x"};
  auto y = Var{"y"};
  auto c = Var{"c"};

  auto f_rescaled = Halide::Func{f.name() + "_rescaled"};
  f_rescaled(x, y, c) = cast<std::uint8_t>(  //
      (f(x, y) - f_min()) /                  //
      (f_max() - f_min()) * 255              //
  );

  return f_rescaled;
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

  auto x = Halide::Var{"x"};
  auto y = Halide::Var{"y"};
  auto c = Halide::Var{"c"};

  auto xi = Halide::Var{"xi"};
  auto yi = Halide::Var{"yi"};

  auto gray = Halide::Func{"gray"};
  {
    auto input_ext = Halide::BoundaryConditions::repeat_edge(input);
    auto r = input_ext(x, y, 0);
    auto g = input_ext(x, y, 1);
    auto b = input_ext(x, y, 2);
    gray(x, y) = (0.2125f * r + 0.7154f * g + 0.0721f * b) / 255.f;
#define USE_SCHEDULE
#ifdef USE_SCHEDULE
    gray.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
    gray.compute_root();
  }

  auto num_scales = 3 + 3;
  auto delta_s = static_cast<float>(std::pow(2, 1./3.));
  auto sigmas = std::vector<float>(num_scales);
  for (auto s = 0; s < num_scales; ++s)
    sigmas[s] = std::pow(delta_s, s + 1);

  auto gauss_pyr = std::vector<hal::GaussianConvolution2D<>>{};
  for (auto s = 0; s < num_scales; ++s)
  {
    auto gauss = hal::GaussianConvolution2D<>{};
    gauss.output = Halide::Func{"gauss_diff"};
    gauss.generate_2d(gray, sigmas[s], 4, frame.width(), frame.height());
    gauss.schedule_2d(Halide::Target{});
    gauss.output.compute_root();

    gauss_pyr.push_back(gauss);
  }

  auto dog_pyr = std::vector<Halide::Func>(num_scales - 1);
  for (auto s = 0; s < num_scales - 1; ++s)
    dog_pyr[s](x, y) =
        gauss_pyr[s + 1].output(x, y) - gauss_pyr[s].output(x, y);

  auto dog_local_max = Halide::Func{"dog_local_max"};
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    dog_local_max(x, y) =
        (Halide::maximum(dog_pyr[0](x + r.x, y + r.y)) == dog_pyr[0](x, y)) * dog_pyr[0](x, y);
  }
#ifdef USE_SCHEDULE
  dog_local_max.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
  dog_local_max.compute_root();

  auto dog_pyr_rgb =
      rescale_to_rgb(dog_local_max, frame.width(), frame.height());
#ifdef USE_SCHEDULE
  dog_pyr_rgb.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
  dog_pyr_rgb.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);

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
      dog_pyr_rgb.realize(output);
    }
    sara::toc("DoG");

    sara::display(dog_frame);
  }

  return 0;
}
