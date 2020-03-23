#define BOOST_TEST_MODULE "CSV I/O"

#include <boost/test/unit_test.hpp>

#include "../shakti_halide_utilities.hpp"


using namespace Halide;


BOOST_AUTO_TEST_CASE(test_gaussian_blur)
{
  // Create the gaussian kernel.
  const auto truncation_factor = 4;
  const auto sigma = 3.f;
  const auto radius = int(sigma / 2) * truncation_factor;
  const auto kernel_size = 2 * radius + 1;

  auto x = Var{"x"};
  auto i = RDom{-radius, kernel_size};

  auto gaussian_unnormalized = Func{"gaussian_unnormalized"};
  gaussian_unnormalized(x) = exp(-(x * x) / (2.f * sigma * sigma));

  auto gaussian_sum = sum(gaussian_unnormalized(i));

  auto gaussian = Func{"gaussian"};
  gaussian(x) = gaussian_unnormalized(x) / gaussian_sum;

  Buffer<float> gaussian_kernel{kernel_size};
  gaussian_kernel.set_min(-radius);
  gaussian.realize(gaussian_kernel);

  SARA_DEBUG << "Gaussian kernel" << std::endl;
  for (auto k = gaussian_kernel.dim(0).min(); k <= gaussian_kernel.dim(0).max();
       ++k)
    std::cout << "g[" << k << "] = " << gaussian_kernel(k) << std::endl;

  auto dirac = Func{"dirac"};

  dirac(x) = select(x == 10, 1.f, 0.f);
  Buffer<float> in = dirac.realize(21);
  auto in_padded = BoundaryConditions::repeat_edge(in);

  auto conv = Func{"conv"};
  conv(x) = sum(in_padded(x + i) * gaussian(i));
  gaussian.compute_root();

  auto conv_custom = Func{"conv_custom"};
  conv_custom(x) = sum(in_padded(x + i) * gaussian_kernel(i));

  Buffer<float> out = conv.realize(21);
  Buffer<float> out_custom = conv_custom.realize(21);

  SARA_DEBUG << "Input" << std::endl;
  for (auto k = 0; k < in.dim(0).extent(); ++k)
    std::cout << k << " " << in(k) << std::endl;

  SARA_DEBUG << "Output" << std::endl;
  for (auto k = 0; k < out.dim(0).extent(); ++k)
    std::cout << k << " " << out(k) << std::endl;

  SARA_DEBUG << "Output custom" << std::endl;
  for (auto k = 0; k < out_custom.dim(0).extent(); ++k)
    std::cout << k << " " << out_custom(k) << std::endl;
}
