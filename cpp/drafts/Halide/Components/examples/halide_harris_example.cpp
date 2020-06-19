#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Utilities.hpp>
#include <drafts/Halide/Components/Differential.hpp>
#include <drafts/Halide/Components/GaussianConvolution2D.hpp>


namespace hal = DO::Shakti::HalideBackend;
namespace sara = DO::Sara;


using namespace std;


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

  auto input = Halide::Buffer<uint8_t>::make_interleaved(
      reinterpret_cast<uint8_t*>(frame.data()), frame.width(), frame.height(),
      3);
  auto gray_r = Halide::Buffer<float>(frame.width(), frame.height());

  auto harris_r = Halide::Buffer<float>(frame.width(), frame.height());

  auto x = Halide::Var{"x"};
  auto y = Halide::Var{"y"};

  auto gray = Halide::Func{"gray"};
  {
    auto input_ext = Halide::BoundaryConditions::repeat_edge(input);
    auto r = input_ext(x, y, 0);
    auto g = input_ext(x, y, 1);
    auto b = input_ext(x, y, 2);
    gray(x, y) = (0.2125f * r + 0.7154f * g + 0.0721f * b) / 255.f;
    gray.compute_root();
  }

  auto gauss_diff = hal::GaussianConvolution2D<>{};
  {
    const auto sigma_d = 1.f;
    // FIXME: gray(x, y, c, n) inside the generator...
    gauss_diff.generate(gray, sigma_d, 4, frame.width(), frame.height());
    gauss_diff.schedule(Halide::Target{});
  }

  auto moment = Halide::Func{"moment"};
  {
    auto gradient_xpr = hal::gradient(gauss_diff.output, x, y);
    auto moment_xpr = gradient_xpr * gradient_xpr.transpose();
    moment(x, y) = moment_xpr;
    moment.compute_root();
  }

  auto gauss_int = std::array<hal::GaussianConvolution2D<>, 4>{};
  {
    const auto sigma_i = 3.f;
    for (int i = 0; i < 4; ++i)
    {
      auto mi = Halide::Func{};
      mi(x, y) = moment(x, y)[i];
      gauss_int[i].generate(mi, sigma_i, 4, frame.width(), frame.height());
      gauss_int[i].schedule(Halide::Target{});
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
  }

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
      //harris.realize(harris_r);
    }
    sara::toc("Harris cornerness function");

    sara::display(frame);
  }
  return 0;
}
