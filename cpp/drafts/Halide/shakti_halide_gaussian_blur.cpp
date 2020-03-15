#include "Halide.h"


namespace {

  using namespace Halide;

  class GaussianBlur : public Halide::Generator<GaussianBlur>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};
    GeneratorParam<float> sigma{"sigma", 10.f};
    GeneratorParam<int> truncation_factor{"trunc", 4};

    Input<Buffer<float>> input{"f", 2};
    Output<Buffer<float>> output{"conv_f", 2};

    Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      const auto w = input.width();
      const auto h = input.height();
      const auto radius = int(sigma / 2) * truncation_factor;

      // Define the unnormalized gaussian function.
      auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
      gaussian_unnormalized(x) = exp(-(x * x) / (2 * sigma * sigma));

      // Define the summation variable `k` defined on a summation domain.
      auto k = RDom(-radius, 2 * radius + 1);
      // Calculate the normalization factor by summing with the summation
      // variable.
      auto normalization_factor = sum(gaussian_unnormalized(k));

      auto gaussian = Func{"gaussian"};
      gaussian(x) = gaussian_unnormalized(x) / normalization_factor;

      // 1st pass: transpose and convolve the columns.
      auto input_t = Func{"input_transposed"};
      input_t(y, x) = input(x, y);

      auto conv_y_t = Func{"conv_y_t"};
      conv_y_t(x, y) = sum(input_t(clamp(x + k, 0, h - 1), y) * gaussian(k));


      // 2nd pass: transpose and convolve the rows.
      auto conv_y = Func{"conv_y"};
      conv_y(x, y) = conv_y_t(y, x);

      auto& conv_x = output;
      conv_x(x, y) = sum(conv_y(clamp(x + k, 0, w - 1), y) * gaussian(k));

      // The CPU schedule.
      {
        // Compute the gaussian first.
        gaussian.compute_root();

        // 1st pass: transpose and convolve the columns
        conv_y_t.compute_root();
        conv_y_t.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);

        // 2nd pass: transpose and convolve the rows.
        conv_y.compute_root();
        conv_x.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);
      }
    }

    void schedule_algorithm()
    {
      // GPU schedule.
      if (get_target().has_gpu_feature())
        output.gpu_tile(x, y, xi, yi, tile_x, tile_y);

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
        output.split(y, yo, yi, 8).parallel(yo).vectorize(x, 8);
    }
  };

}  // namespace

HALIDE_REGISTER_GENERATOR(GaussianBlur, shakti_halide_gaussian_blur)
