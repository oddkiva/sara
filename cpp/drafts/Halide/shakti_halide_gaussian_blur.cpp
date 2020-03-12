#include "Halide.h"


namespace {

  using namespace Halide;


  class GaussianBlur : public Halide::Generator<GaussianBlur>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};
    GeneratorParam<float> sigma{"sigma", 3.f};
    GeneratorParam<int> truncation_factor{"trunc", 4};

    Input<Buffer<float>> input{"f", 2};
    Output<Buffer<float>> output{"conv_f", 2};

    Var x{"x"}, y{"y"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      const auto kernel_size = int(sigma / 2) * truncation_factor + 1;

      auto gaussian = Func{"gaussian"};
      gaussian(x) = exp(-(x * x) / (2 * sigma * sigma));

      auto r = RDom(-kernel_size / 2, kernel_size / 2);
      gaussian(x) /= sum(gaussian(r));

      auto input_padded = BoundaryConditions::repeat_edge(input);

      auto conv_x = Func{"conv_x"};
      conv_x(x, y) = sum(input_padded(x + r, y) * gaussian(r));

      auto conv_x_padded = BoundaryConditions::repeat_edge(conv_x);
      output(x, y) = sum(conv_x_padded(x, y + r) * gaussian(r));

      gaussian.compute_root();

      schedule_algorithm();
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
            .split(y, y, yi, 128)
            .parallel(y)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
        output.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(RgbToGray, shakti_halide_rgb_to_gray)
