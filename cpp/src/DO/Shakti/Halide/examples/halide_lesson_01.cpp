#include "Halide.h"


auto main() -> int
{
  auto gradient = Halide::Func{};

  auto x = Halide::Var{};
  auto y = Halide::Var{};

  auto e = x + y;
  gradient(x, y) = e;

  Halide::Buffer<int32_t> output = gradient.realize(800, 600);

  for (auto v = 0; v < output.height(); ++v)
    for (auto u = 0; u < output.width(); ++u)
      if (output(u, v) != u + v)
        printf("Something went wrong!");

  printf("Success!\n");

  return 0;
}
