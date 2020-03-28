#include "Halide.h"


namespace {

  using namespace Halide;

  class Enlarge : public Halide::Generator<Enlarge>
  {
  public:
    GeneratorParam<int> tile_x{"tile_x", 8};
    GeneratorParam<int> tile_y{"tile_y", 8};

    Input<Buffer<float>> input{"input", 3};
    Input<int[2]> in_sizes{"in_sizes"};
    Input<int[2]> out_sizes{"out_sizes"};
    Output<Buffer<float>> output{"enlarged_input", 3};

    Var x{"x"}, y{"y"}, c{"c"};
    Var xo{"xo"}, yo{"yo"}, co{"co"};
    Var xi{"xi"}, yi{"yi"}, ci{"ci"};

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

      output(x, y, c) = (1 - wx) * (1 - wy) * input_padded(xr, yr, c) +  //
                        wx * (1 - wy) * input_padded(xr + 1, yr, c) +    //
                        (1 - wx) * wy * input_padded(xr, yr + 1, c) +    //
                        wx * wy * input_padded(xr + 1, yr + 1, c);
    }

    void schedule()
    {
      input.dim(0).set_stride(Expr());  // Use an undefined Expr to
                                        // mean there is no
                                        // constraint.

      output.dim(0).set_stride(Expr());

      Expr input_is_planar = input.dim(0).stride() == 1;
      Expr input_is_interleaved = input.dim(0).stride() == 3 and  //
                                  input.dim(2).stride() == 1 and  //
                                  input.dim(2).extent() == 3;

      Expr output_is_planar = output.dim(0).stride() == 1;
      Expr output_is_interleaved = output.dim(0).stride() == 3 and  //
                                   output.dim(2).stride() == 1 and  //
                                   output.dim(2).extent() == 3;

      // GPU schedule.
      if (get_target().has_gpu_feature())
      {
        output.specialize(input_is_planar && output_is_planar)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 1);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .gpu_tile(x, y, c, xo, yo, co, xi, yi, ci, tile_x, tile_y, 3);
      }

      // Hexagon schedule.
      else if (get_target().features_any_of({Target::HVX_64, Target::HVX_128}))
      {
        const auto vector_size =
            get_target().has_feature(Target::HVX_128) ? 128 : 64;

        output.specialize(input_is_planar && output_is_planar)
            .hexagon()
            .prefetch(input, y, 2)
            .split(y, yo, yi, 128)
            .parallel(yo)
            .vectorize(x, vector_size);
      }

      // CPU schedule.
      else
      {
        output.specialize(input_is_planar && output_is_planar)
            .split(y, yo, yi, 8)
            .parallel(yo)
            .vectorize(x, 8);

        output.specialize(input_is_interleaved && output_is_interleaved)
            .reorder(c, x, y)
            .unroll(c);
      }
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(Enlarge, shakti_halide_enlarge)
