#define BOOST_TEST_MODULE "Halide/Image Pyramid"

#include <boost/test/unit_test.hpp>
#include <type_traits>

#include <drafts/Halide/ImagePyramid.hpp>


BOOST_AUTO_TEST_CASE(test_image_pyramid)
{
  namespace halide = DO::Shakti::HalideBackend;
  auto pyramid = halide::ImagePyramid<float>{};

  const auto num_octaves = 2;
  const auto num_scales_per_octave = 3;
  const auto sigma_initial = 1.6f;
  const auto scale_geometric_factor = std::pow(2.f, 1.f / 3.f);

  const auto hw_sizes = Eigen::Vector2i{768, 1024};

  pyramid.reset(hw_sizes, num_octaves, num_scales_per_octave, sigma_initial,
                scale_geometric_factor);

  BOOST_CHECK_EQUAL(num_octaves, pyramid.num_octaves());
  BOOST_CHECK_EQUAL(num_scales_per_octave, pyramid.num_scales_per_octave());
  BOOST_CHECK_EQUAL(sigma_initial, pyramid.scale_initial());
  BOOST_CHECK_EQUAL(scale_geometric_factor, pyramid.scale_geometric_factor());

  for (int i = 0; i < pyramid.num_octaves(); ++i)
    BOOST_CHECK_EQUAL(pyramid(i).size(0),
                      static_cast<size_t>(pyramid.num_scales_per_octave()));
}
