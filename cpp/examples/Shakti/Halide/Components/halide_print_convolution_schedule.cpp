#include <Halide.h>

auto main() -> int
{
  auto x = Halide::Var{"x"};
  auto y = Halide::Var{"y"};

  auto xo = Halide::Var{"xo"};
  auto yo = Halide::Var{"yo"};

  auto xi = Halide::Var{"xi"};
  auto yi = Halide::Var{"yi"};
  auto tile = Halide::Var{"tile"};

  auto input = Halide::Func{"input"};
  auto kernel = Halide::Func{"kernel"};

  input(x, y) = x + y;
  kernel(x) = Halide::exp(-x);

  auto conv_x = Halide::Func{"conv_x"};
  auto conv_y = Halide::Func{"conv_y"};

  const auto ksz = 20;
  auto k = Halide::RDom(0, ksz, "k");

  // The algorithm
  conv_x(x, y) = Halide::sum(input(x + k - ksz / 2, y) * kernel(k), "conv_x");
  conv_y(x, y) = Halide::sum(conv_x(x, y + k - ksz / 2) * kernel(k), "conv_y");

  // The schedule
  kernel.compute_root();

  conv_y  //
      .tile(x, y, xo, yo, xi, yi, 64, 64)
      .fuse(xo, yo, tile)
      .parallel(tile)
      // .parallel(yo)
      .vectorize(xi, 4, Halide::TailStrategy::GuardWithIf)  //
      ;
  conv_x  //
          // .store_at(conv_y, tile)
      .compute_at(conv_y, xi)                               //
      .vectorize(x, 4, Halide::TailStrategy::GuardWithIf)  //
      ;

  conv_y.print_loop_nest();
  conv_y.compile_to_lowered_stmt("separable_conv_2d.stmt.html", {},
                                 Halide::HTML);
  conv_y.compile_to_assembly("separable_conv_2d.s", {});

  return 0;
}
