#include "Halide.h"


namespace {

  using namespace Halide;

  class Enlarge : public Halide::Generator<Enlarge>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 32};
    GeneratorParam<int> tile_y{"tile_y", 32};

    Input<Buffer<float>> input{"input", 2};
    Input<int[2]> in_sizes{"in_sizes"};
    Input<int[2]> out_sizes{"out_sizes"};
    Output<Buffer<float>> output{"enlarged_input", 2};

    Var x{"x"}, y{"y"}, xo{"xo"}, yo{"yo"}, xi{"xi"}, yi{"yi"};

    void generate()
    {
      Expr w_in = cast<float>(in_sizes[0]);
      Expr h_in = cast<float>(in_sizes[1]);
      Expr w_out = cast<float>(out_sizes[0]);
      Expr h_out = cast<float>(out_sizes[1]);

      Expr xx = x * w_in / w_out;
      Expr yy = y * h_in / h_out;

      auto input_padded = BoundaryConditions::repeat_edge(input);

      auto wx = xx - floor(xx);
      auto wy = yy - floor(yy);

      auto xr = cast<int>(xx);
      auto yr = cast<int>(yy);

      output(x, y) = (1 - wx) * (1 - wy) * input_padded(xr, yr) +  //
                     wx * (1 - wy) * input_padded(xr + 1, yr) +    //
                     (1 - wx) * wy * input_padded(xr, yr + 1) +    //
                     wx * wy * input_padded(xr + 1, yr + 1);

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
