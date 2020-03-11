#include "Halide.h"


namespace {

  using namespace Halide;


  class RgbToGray : public Halide::Generator<RgbToGray>
  {
  public:
    Input<Buffer<std::uint8_t>> input{"Rgb", 3};
    Output<Buffer<float>> output{"Gray", 2};

    void generate()
    {
      // Deal with interleaved RGB pixel format.
      input.dim(0).set_stride(3).dim(2).set_stride(1);
      input.dim(2).set_bounds(0, 3);

      Var x{"x"}, y{"y"};

      auto r = input(x, y, 0) / 255.f;
      auto g = input(x, y, 1) / 255.f;
      auto b = input(x, y, 2) / 255.f;

      auto gray = Func{"gray"};
      output(x, y) = 0.2125f * r + 0.7154f * g + 0.0721f * b;
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(RgbToGray, sara_halide_rgb_to_gray)
