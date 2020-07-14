#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/GaussianConvolution2D.hpp>
#include <drafts/Halide/Components/LocalExtremum.hpp>
#include <drafts/Halide/Utilities.hpp>

#define USE_SCHEDULE


namespace hal = DO::Shakti::HalideBackend;
namespace sara = DO::Sara;


using namespace std;


auto bool_to_rgb(const Halide::Func& f)
{
  auto x = Halide::Var{"x"};
  auto y = Halide::Var{"y"};
  auto yi = Halide::Var{"yi"};
  auto c = Halide::Var{"c"};

  auto f_rescaled = Halide::Func{f.name() + "_rescaled"};
  f_rescaled(x, y, c) = Halide::cast<std::uint8_t>(f(x, y)) * 255;
#ifdef USE_SCHEDULE
  f_rescaled.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
  f_rescaled.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);

  return f_rescaled;
}

auto rescale_to_rgb(const Halide::Func& f, int32_t w, int32_t h)
{
  auto r = Halide::RDom(0, w, 0, h);
  auto f_min = Halide::Func{f.name() + "_max"};
  auto f_max = Halide::Func{f.name() + "_min"};
  f_max() = Halide::maximum(f(r.x, r.y));
  f_min() = Halide::minimum(f(r.x, r.y));
  f_max.compute_root();
  f_min.compute_root();

  auto x = Halide::Var{"x"};
  auto y = Halide::Var{"y"};
  auto c = Halide::Var{"c"};

  auto f_rescaled = Halide::Func{f.name() + "_rescaled"};
  f_rescaled(x, y, c) = Halide::cast<std::uint8_t>(  //
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
  auto harris_frame = sara::Image<sara::Rgb8>{frame.sizes()};

  auto input = hal::as_interleaved_buffer(frame);
  auto output = hal::as_interleaved_buffer(harris_frame);

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
#ifdef USE_SCHEDULE
    gray.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
    gray.compute_root();
  }

  auto gauss_diff = hal::GaussianConvolution2D<>{};
  {
    const auto sigma_d = 0.5f;
    gauss_diff.output = Halide::Func{"gauss_diff"};
    gauss_diff.generate_2d(gray, sigma_d, 4, frame.width(), frame.height());
    gauss_diff.schedule_2d(Halide::Target{});
    gauss_diff.output.compute_root();
  }

  auto moment = Halide::Func{"moment"};
  {
    auto gradient_xpr = hal::gradient(gauss_diff.output, x, y);
    auto moment_xpr = gradient_xpr * gradient_xpr.transpose();
    moment(x, y) = moment_xpr;
#ifdef USE_SCHEDULE
    moment.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
    moment.compute_root();
  }

  auto gauss_int = std::array<hal::GaussianConvolution2D<>, 4>{};
  {
    const auto sigma_i = 1.f;
    for (int i = 0; i < 4; ++i)
    {
      auto mi = Halide::Func{"moment_" + std::to_string(i)};
      mi(x, y) = moment(x, y)[i];
      gauss_int[i].output = Halide::Func{"gauss_int_" + std::to_string(i)};
      gauss_int[i].generate_2d(mi, sigma_i, 4, frame.width(), frame.height());
      gauss_int[i].schedule_2d(Halide::Target{});
      gauss_int[i].output.compute_root();
    }
  }

  const auto kappa = 4.f;
  auto harris = Halide::Func{};
  {
    auto m = hal::Matrix<2, 2>{};
    m(0, 0) = gauss_int[0].output(x, y);
    m(0, 1) = gauss_int[1].output(x, y);
    m(1, 0) = gauss_int[2].output(x, y);
    m(1, 1) = gauss_int[3].output(x, y);
    harris(x, y) = hal::det(m) - kappa * Halide::pow(hal::trace(m), 2);
#ifdef USE_SCHEDULE
    harris.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
    harris.compute_root();
  }


  auto harris_local_max = Halide::Func{"harris_local_max"};
  {
    auto r = Halide::RDom{-1, 3, -1, 3};
    harris_local_max(x, y) =
        Halide::maximum(harris(x + r.x, y + r.y)) == harris(x, y);
  }
  // auto harris_local_max= hal::local_max(harris, x, y);
#ifdef USE_SCHEDULE
  harris_local_max.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
  harris_local_max.compute_root();

  auto harris_local_max_rgb = bool_to_rgb(harris_local_max);

  auto harris_rescaled = rescale_to_rgb(harris, frame.width(), frame.height());
#ifdef USE_SCHEDULE
  harris_rescaled.split(y, y, yi, 4).parallel(y).vectorize(x, 8);
#endif
  harris_rescaled.output_buffer()
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
      //harris_rescaled.realize(output);
      harris_local_max_rgb.realize(output);
    }
    sara::toc("Harris cornerness function");

    sara::display(harris_frame);
  }
  return 0;
}
