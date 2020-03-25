#include "Halide.h"


namespace {

  using namespace Halide;

  template <typename T>
  class Enlarge : public Halide::Generator<Enlarge<T>>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<T>> input{"f", 2};
    Output<Buffer<T>> output{"g", 2};

    Var x{"x"}, y{"y"}, c{"c"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      const float w_in = input.width();
      const float h_in = input.height();

      const float w_out = input.width();
      const float h_out = input.height();

      auto xx = x * w_in / w_out;
      auto yy = y * h_in / h_out;

      auto weights = Func{"weights"};
      weights(x, y) = abs(1 - xx - x) * abs(1 - yy - y);

      auto input_padded = BoundaryConditions::repeat_edge(input);

      auto r = RDom{floor(xx), 2, floor(yy), 2};
      output(x, y, c) = sum(weights(r) * input_padded(r.x, r.y, c));

      // GPU schedule.
      if (get_target().has_gpu_feature())
        output.gpu_tile(x, y, xo, yo, xi, yi, tile_x, tile_y);

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

HALIDE_REGISTER_GENERATOR(Enlarge, shakti_halide_enlarge)
