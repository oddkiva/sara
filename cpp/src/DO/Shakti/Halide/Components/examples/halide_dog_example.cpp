#include <DO/Sara/Core.hpp>
#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/Components/Differential.hpp>
#include <DO/Shakti/Halide/Components/DoGExtremum.hpp>
#include <DO/Shakti/Halide/Components/GaussianConvolution2D.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>


namespace hal = DO::Shakti::HalideBackend;
namespace sara = DO::Sara;


struct DoGExtrema
{
  std::vector<int32_t> x;
  std::vector<int32_t> y;
  std::vector<std::int8_t> type;

  auto resize(std::size_t size)
  {
    x.resize(size);
    y.resize(size);
    type.resize(size);
  }
};

struct DoGResiduals
{
  std::vector<float> dx;
  std::vector<float> dy;
  std::vector<float> ds;
  std::vector<float> values;
  std::vector<std::uint8_t> successes;

  auto resize(std::size_t size)
  {
    dx.resize(size);
    dy.resize(size);
    ds.resize(size);
    values.resize(size);
    successes.resize(size);
  }
};


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


auto convert_to_grayscale(const Halide::Func& f)
{
  auto gray = Halide::Func{f.name() + "_grayscaled"};
  auto r = f(x, y, 0);
  auto g = f(x, y, 1);
  auto b = f(x, y, 2);
  gray(x, y) = (0.2125f * r + 0.7154f * g + 0.0721f * b) / 255.f;
  return gray;
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

  auto f_rescaled = Halide::Func{f.name() + "_rescaled"};
  f_rescaled(x, y, c) = Halide::cast<std::uint8_t>(  //
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

auto schedule_rescale_rgb(const Halide::Func& f, int w, int h, bool use_gpu,
                          const Halide::Target& jit_target)
{
  auto output_as_rgb = rescale_to_rgb(f, w, h);
  schedule(output_as_rgb, use_gpu);
  output_as_rgb.output_buffer()
      .dim(0)
      .set_stride(3)
      .dim(2)
      .set_stride(1)
      .set_bounds(0, 3);
  output_as_rgb.compile_jit(jit_target);
  return output_as_rgb;
}

auto schedule_dog_extremum_reweighted(const Halide::Func& dog_extremum,
                                      const std::vector<Halide::Func>& dog_pyr,
                                      bool use_gpu,
                                      const Halide::Target& jit_target,
                                      int scale_index = 1)
{
  auto dog_extremum_reweighted = Halide::Func{"dog_extremum_reweighted"};
  dog_extremum_reweighted(x, y) =
      select(dog_extremum(x, y) != 0, dog_pyr[scale_index](x, y), 0.f);
  schedule(dog_extremum_reweighted, use_gpu);
  dog_extremum_reweighted.compile_jit(jit_target);
  return dog_extremum_reweighted;
}

auto schedule_extremum_count(const Halide::Func& extremum_map, int w, int h,
                             const Halide::Target& jit_target)
{
  auto extremum_count_fn = Halide::Func{extremum_map.name() + "_count"};
  extremum_count_fn() = hal::count_extrema(extremum_map, w, h);
  extremum_count_fn.compile_jit(jit_target);
  return extremum_count_fn;
}

auto schedule_gaussian_octave(const Halide::Func& gray,
                              const std::vector<float>& sigmas, int w, int h,
                              const Halide::Target& jit_target)
{
  auto gauss_pyr = std::vector<hal::GaussianConvolution2D<>>(sigmas.size());
  for (auto s = 0u; s < sigmas.size(); ++s)
  {
    const auto scale_factorigma =
        s == 0 ? sigmas[0]
               : std::sqrt(std::pow(sigmas[s], 2) - std::pow(sigmas[s - 1], 2));

    const auto& prev = s == 0u ? gray : gauss_pyr[s - 1].output;

    gauss_pyr[s].output = Halide::Func{"gauss_" + std::to_string(s)};
    gauss_pyr[s].generate_2d(prev, scale_factorigma, 4, w, h);
    gauss_pyr[s].schedule_2d(jit_target);
    gauss_pyr[s].output.compute_root();
  }

  return gauss_pyr;
}

auto schedule_subtraction(
    const std::vector<hal::GaussianConvolution2D<>>& gauss_pyr,
    bool use_gpu)
{
  auto dog_pyr = std::vector<Halide::Func>(gauss_pyr.size() - 1);
  for (auto s = 0u; s < gauss_pyr.size() - 1; ++s)
  {
    dog_pyr[s] = Halide::Func{"dog_pyr_" + std::to_string(s)};
    dog_pyr[s](x, y) = gauss_pyr[s + 1].output(x, y) -  //
                       gauss_pyr[s].output(x, y);       //
    schedule(dog_pyr[s], use_gpu);
  }
  return dog_pyr;
}


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath = "/Users/david/Desktop/Datasets/videos/sample1.mp4"s;
#else
  // const auto video_filepath = "/home/david/Desktop/test.mp4"s;
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto dog_map = sara::Image<std::int8_t>{frame.sizes()};
  auto dog_extrema = DoGExtrema{};
  auto dog_residuals = DoGResiduals{};
  // Fix a hard upper-bound.
  dog_extrema.resize(4096);
  dog_residuals.resize(4096);

  // Hardware-specific scheduling.
  constexpr auto use_gpu = false;
  const auto jit_target = use_gpu ? hal::get_gpu_target()    //
                                  : hal::get_host_target();  //

  // Halide pipeline buffers.
  auto frame_buffer = hal::as_interleaved_buffer(frame);
  auto dog_map_buffer = hal::as_buffer(dog_map);
  auto dog_x = hal::as_buffer(dog_extrema.x);
  auto dog_y = hal::as_buffer(dog_extrema.y);
  auto dog_type = hal::as_buffer(dog_extrema.type);
  auto dog_res_x = hal::as_buffer(dog_residuals.dx);
  auto dog_res_y = hal::as_buffer(dog_residuals.dy);
  auto dog_res_s = hal::as_buffer(dog_residuals.ds);
  auto dog_values = hal::as_buffer(dog_residuals.values);
  auto dog_successes = hal::as_buffer(dog_residuals.successes);

  // 0. Extend the domain of the input image.
  auto frame_ext = Halide::BoundaryConditions::repeat_edge(frame_buffer);

  // 1. Conversion to grayscale float.
  auto gray = convert_to_grayscale(frame_ext);
  gray.compute_root();

  // 2. Gaussian octave scales.
  const auto num_scales = 3 + 3;
  const auto scale_factor = static_cast<float>(std::pow(2, 1. / 3.));
  auto sigmas = std::vector<float>(num_scales);
  for (auto s = 0; s < num_scales; ++s)
    sigmas[s] = std::pow(scale_factor, s + 1);

  // 3. Gaussian octave.
  auto gauss_pyr = schedule_gaussian_octave(gray,                           //
                                            sigmas,                         //
                                            frame.width(), frame.height(),  //
                                            jit_target);                    //
  for (auto& gauss : gauss_pyr)
    gauss.output.compute_root();

  // 4. DoG octave.
  auto dog_pyr = schedule_subtraction(gauss_pyr, use_gpu);
  for(auto& dog: dog_pyr)
    dog.compute_root();

  // 5. Local scale-space extrema.
  auto dog_extremum_map = Halide::Func{"dog_extremum_map"};
  {
    const Halide::Expr edge_ratio = 10.f;
    const Halide::Expr extremum_thres = 0.03f;
    dog_extremum_map(x, y) = hal::is_dog_extremum(  //
        dog_pyr[0], dog_pyr[1], dog_pyr[2],         //
        edge_ratio, extremum_thres,                 //
        x, y);                                      //
  }
  schedule(dog_extremum_map, use_gpu);
  dog_extremum_map.compute_root();
  dog_extremum_map.compile_jit(jit_target);

  // Count extrema.
  auto dog_extremum_count_fn = schedule_extremum_count(  //
      dog_extremum_map,                                  //
      frame.width(), frame.height(),                     //
      jit_target);                                       //

  // 6. Residual estimation.
  auto dog_residuals_fn = Halide::Func{"dog_extremum_residuals"};
  {
    auto i = Halide::Var{"i"};
    auto ii = Halide::Var{"ii"};
    const auto tile_i = 32;

    const auto x0 = Halide::clamp(Halide::cast<int32_t>(dog_x(i)),  //
                                  0, frame.width());                //
    const auto y0 = Halide::clamp(Halide::cast<int32_t>(dog_y(i)),  //
                                  0, frame.height());               //
    const auto r = hal::refine_extremum_v1(                         //
        dog_pyr[0], dog_pyr[1], dog_pyr[2],                         //
        x0, y0);                                                    //

    dog_residuals_fn(i) = {r[0], r[1], r[2], r[3],
                           Halide::cast<std::uint8_t>(r[4])};
    if (use_gpu)
      dog_residuals_fn.gpu_tile(i, ii, tile_i,
                                Halide::TailStrategy::GuardWithIf);
    else
      dog_residuals_fn.vectorize(i, 8);
    dog_residuals_fn.compile_jit(jit_target);
  }


  sara::create_window(frame.sizes());
  sara::set_antialiasing(sara::active_window());
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
      frame_buffer.set_host_dirty();
      dog_extremum_map.realize(dog_map_buffer);
      dog_map_buffer.copy_to_host();
    }
    sara::toc("DoG");

    sara::tic();
    const auto num_dog_extrema = std::count_if(
        dog_map.begin(), dog_map.end(), [](const auto& v) { return v != 0; });
    SARA_CHECK(num_dog_extrema);
    sara::toc("Extremum count");

    // Fill the arrays of DoG extrema.
    sara::tic();
    {
      auto dog_index = 0;
      for (auto y = 0; y < frame.height(); ++y)
      {
        for (auto x = 0; x < frame.width(); ++x)
        {
          if (dog_map(x, y) == 0)
            continue;

          dog_extrema.x[dog_index] = x;
          dog_extrema.y[dog_index] = y;
          dog_extrema.type[dog_index] = dog_map(x, y);
          ++dog_index;
        }
      }
      if (dog_index != num_dog_extrema)
        throw std::runtime_error{"Error counting DoG extrema!"};
    }
    sara::toc("DoG Array filling");

    sara::tic();
    {
      // Inputs.
      dog_x.set_host_dirty();
      dog_y.set_host_dirty();
      // The operation.
      dog_residuals_fn.realize({dog_res_x, dog_res_y, dog_res_s,  //
                                dog_values,                       //
                                dog_successes});                  //
      // Outputs.
      dog_res_x.copy_to_host();
      dog_res_y.copy_to_host();
      dog_res_s.copy_to_host();
      dog_values.copy_to_host();
      dog_successes.copy_to_host();
    }
    sara::toc("DoG localization residuals");

    sara::tic();
    {
      sara::display(frame);
      auto num_residual_successes = int{};
      for (auto i = 0; i < num_dog_extrema; ++i)
      {
        const auto color = dog_type(i) == 1 ? sara::Blue8 : sara::Red8;

        const auto& x = dog_x(i);
        const auto& y = dog_y(i);
        const auto& s = sigmas[1];

        const auto& res_x = dog_res_x(i);
        const auto& res_y = dog_res_y(i);
        const auto& res_s = dog_res_s(i);

        const auto x1 = dog_successes(i) ? x + res_x : x;
        const auto y1 = dog_successes(i) ? y + res_y : y;
        const auto s1 = dog_successes(i)
                            ? s * std::pow(scale_factor, res_s)
                            : s;

        num_residual_successes += dog_successes(i);
        sara::draw_circle(x1, y1, static_cast<int>(s1 * std::sqrt(2.f)), color,
                          2);
      }

      // SARA_CHECK(num_residual_successes);
      if (num_residual_successes > num_dog_extrema)
        throw std::runtime_error{"Error calculating localization residuals!"};
    }
    sara::toc("Visualization");
  }

  return 0;
}
