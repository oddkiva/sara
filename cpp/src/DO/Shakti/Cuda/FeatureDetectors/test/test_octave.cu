// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

// Convolving in batch the input image does not seem very fast.
// Rather convolving sequentially seems much faster if we base ourselves from
// the computation time spent in the Halide implementation.


#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Octave"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Cuda/Utilities.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;


auto copy(sara::ImageView<float, 3>& src, sc::Octave<float>& dst)
{
  if (src.width() != dst.width() ||    //
      src.height() != dst.height() ||  //
      src.depth() != dst.scale_count())
    throw std::runtime_error{"Invalid sizes!"};

  auto copy_params = cudaMemcpy3DParms{};
  {
    copy_params.srcPtr = make_cudaPitchedPtr(                   //
        reinterpret_cast<void*>(src.data()),                    //
        src.width() * sizeof(float), src.width(), src.height()  //
    );
    copy_params.srcPos = make_cudaPos(0, 0, 0);

    copy_params.dstArray = dst;
    copy_params.dstPos = make_cudaPos(0, 0, 0);

    // Because we use a CUDA array the extent is in terms of number of elements
    // and not in bytes.
    copy_params.extent =
        make_cudaExtent(src.width(), src.height(), src.depth());
    copy_params.kind = cudaMemcpyHostToDevice;
  }

  SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&copy_params));
}

auto copy(sc::Octave<float>& src, sara::ImageView<float, 3>& dst)
{
  if (src.width() != dst.width() ||    //
      src.height() != dst.height() ||  //
      src.scale_count() != dst.depth())
    throw std::runtime_error{"Invalid sizes!"};

  auto copy_params = cudaMemcpy3DParms{};
  {
    copy_params.srcArray = src;
    copy_params.srcPos = make_cudaPos(0, 0, 0);
    copy_params.dstPtr = make_cudaPitchedPtr(                   //
        reinterpret_cast<void*>(dst.data()),                    //
        dst.width() * sizeof(float), dst.width(), dst.height()  //
    );
    copy_params.dstPos = make_cudaPos(0, 0, 0);

    // Because we use a CUDA array the extent is in terms of number of elements
    // and not in bytes.
    copy_params.extent =
        make_cudaExtent(src.width(), src.height(), src.scale_count());
    copy_params.kind = cudaMemcpyDeviceToHost;
  }

  SHAKTI_SAFE_CUDA_CALL(cudaMemcpy3D(&copy_params));
}

BOOST_AUTO_TEST_CASE(test_octave_with_different_data_types)
{
  static constexpr auto w = 1920;
  static constexpr auto h = 1080;
  static constexpr auto scale_count = 3;

  const auto octave_16u =
      sc::make_gaussian_octave<std::uint16_t>(w, h, scale_count);
  const auto octave_16f = sc::make_gaussian_octave<half>(w, h, scale_count);
  const auto octave_32f = sc::make_gaussian_octave<half>(w, h, scale_count);

  // DOES NOT WORK.
  // auto octave_64f = DO::Shakti::Cuda::make_gaussian_octave<double>(w, h);
  BOOST_CHECK_EQUAL(octave_16u.scale_count(), 6);
  BOOST_CHECK_EQUAL(octave_16f.scale_count(), 6);
}

BOOST_AUTO_TEST_CASE(test_copy)
{
  static constexpr auto w = 11;
  static constexpr auto h = 11;
  static constexpr auto scale_count = 1;

  // Initialize the octave CUDA surface.
  auto octave_32f = sc::make_gaussian_octave<float>(w, h, scale_count);
  octave_32f.init_surface();
  BOOST_CHECK_EQUAL(octave_32f.scale_count(), scale_count + 3);

  auto values = sara::Image<float, 3>{w, h, octave_32f.scale_count()};
  values.flat_array().fill(1);

  auto values2 = sara::Image<float, 3>{w, h, octave_32f.scale_count()};
  values2.flat_array().fill(0);

  copy(values, octave_32f);
  copy(octave_32f, values2);
  BOOST_CHECK(values == values2);
}


__global__ void fill(cudaSurfaceObject_t output, int width, int height,
                     int scale_count)
{
  // Calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < scale_count)
  {
    const float out = z * width * height + y * width + x;
    surf2DLayeredwrite(out, output, x * sizeof(float), y, z);
  }
}

BOOST_AUTO_TEST_CASE(test_fill)
{
  static constexpr auto w = 3;
  static constexpr auto h = 5;
  static constexpr auto scale_count = 1;

  // Initialize the octave CUDA surface.
  auto octave = sc::make_gaussian_octave<float>(w, h, scale_count);
  octave.init_surface();

  // Initialize the octave in CUDA.
  const dim3 threadsperBlock(16, 16, 2);
  const dim3 numBlocks(
      (octave.width() + threadsperBlock.x - 1) / threadsperBlock.x,
      (octave.height() + threadsperBlock.y - 1) / threadsperBlock.y,
      (octave.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
  fill<<<numBlocks, threadsperBlock>>>(octave.surface_object(),  //
                                       octave.width(), octave.height(),
                                       octave.scale_count()  //
  );

  // Check the result.
  auto result = sara::Image<float, 3>{w, h, octave.scale_count()};
  result.flat_array().fill(0);
  copy(octave, result);

  auto gt = sara::Image<float, 3>{w, h, octave.scale_count()};
  for (auto i = 0u; i < gt.size(); ++i)
    gt.data()[i] = i;

  // for (auto s = 0; s < result.depth(); ++s)
  //   SARA_DEBUG << s << "\n"
  //              << sara::tensor_view(result)[s].matrix() << std::endl;
  BOOST_CHECK(result == gt);
}


static constexpr auto max_thread_count = 1024;
static constexpr auto tile_x = 32;
static constexpr auto tile_y = 32;
static constexpr auto tile_z = max_thread_count / tile_x / tile_y;

__constant__ float constant_gauss_kernels[512];
__constant__ int constant_gauss_kernel_sizes[16];
__constant__ int constant_gauss_kernel_radius;
__constant__ int constant_kernel_count;
__constant__ int constant_kernel_size;


__global__ void convolve_x(cudaSurfaceObject_t input,   //
                           cudaSurfaceObject_t output,  //
                           int input_layer,             //
                           int width, int height,       //
                           int scale_count)
{
  // Calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < scale_count)
  {
    auto out = float{};

#pragma unroll
    for (auto k = 0; k <= constant_kernel_size; ++k)
    {
      float val;
      surf2DLayeredread(                                           //
          &val,                                                    //
          input,                                                   //
          (x - constant_gauss_kernel_radius + k) * sizeof(float),  //
          y,                                                       //
          input_layer,                                             //
          cudaBoundaryModeClamp);
      out += constant_gauss_kernels[z * constant_kernel_size + k] * val;
    }

    surf2DLayeredwrite<float>(out, output, x * sizeof(float), y, z);
  }
}

__global__ void convolve_y(cudaSurfaceObject_t input,   //
                           cudaSurfaceObject_t output,  //
                           int width, int height,       //
                           int scale_count)
{
  // Calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < scale_count)
  {
    auto out = float{};

#pragma unroll
    for (auto k = 0; k <= constant_kernel_size; ++k)
    {
      float val;
      surf2DLayeredread(                         //
          &val,                                  //
          input,                                 //
          x * sizeof(float),                     //
          y - constant_gauss_kernel_radius + k,  //
          z,                                     //
          cudaBoundaryModeClamp);
      out += constant_gauss_kernels[z * constant_kernel_size + k] * val;
    }

    surf2DLayeredwrite<float>(out, output, x * sizeof(float), y, z);
  }
}

BOOST_AUTO_TEST_CASE(test_convolve)
{
  std::cout << shakti::get_devices().front() << std::endl;

  static constexpr auto scale_count = 3;
  static constexpr auto scale_camera = 1.f;
  static constexpr auto scale_initial = 1.6f;
  static constexpr auto gaussian_truncation_factor = 4.f;
  static const float scale_factor = std::pow(2.f, 1.f / scale_count);

  // Set up the list of scales in the discrete octave.
  auto scales = std::vector<float>(scale_count + 3);
  for (auto i = 0; i < scale_count + 3; ++i)
    scales[i] = scale_initial * std::pow(scale_factor, i);

  // Calculate the Gaussian smoothing values.
  auto sigmas = std::vector<float>(scale_count + 3);
  for (auto i = 0u; i < sigmas.size(); ++i)
    sigmas[i] = std::sqrt(std::pow(scales[i], 2) - std::pow(scale_camera, 2));

  auto sigmaDeltas = std::vector<float>(scale_count + 3);
  for (auto i = 0u; i < sigmas.size(); ++i)
    sigmaDeltas[i] =
        i == 0 ? std::sqrt(std::pow(scales[0], 2) - std::pow(scale_camera, 2))
               : std::sqrt(std::pow(scales[i], 2) - std::pow(scales[i - 1], 2));
  SARA_CHECK(
      Eigen::Map<const Eigen::RowVectorXf>(sigmas.data(), sigmas.size()));
  SARA_CHECK(Eigen::Map<const Eigen::RowVectorXf>(sigmaDeltas.data(),
                                                  sigmaDeltas.size()));

  // Calculater the kernel dimensions.
  auto kernel_sizes = std::vector<int>{};
  for (const auto& sigma : sigmas)
  {
    const auto radius = static_cast<int>(               //
        std::round(gaussian_truncation_factor * sigma)  //
    );
    kernel_sizes.push_back(2 * radius + 1);
  }

  const auto kernel_size_max = kernel_sizes.back();
  const auto kernel_radius = kernel_size_max / 2;

  // Fill the Gaussian kernels.
  auto kernels = sara::Tensor_<float, 2>{
      scale_count + 3,  //
      kernel_size_max   //
  };
  kernels.flat_array().fill(0);

  for (auto n = 0; n < kernels.size(0); ++n)
  {
    const auto& sigma = sigmas[n];
    const auto ksize = kernel_sizes[n];
    const auto kradius = ksize / 2;
    const auto two_sigma_squared = 2 * sigma * sigma;

    for (auto k = 0; k < ksize; ++k)
      kernels(n, k + kernel_radius - kradius) =
          exp(-std::pow(k - kradius, 2) / two_sigma_squared);

    const auto kernel_sum =
        std::accumulate(&kernels(n, kernel_radius - kradius),
                        &kernels(n, kernel_radius - kradius) + ksize, 0.f);

    for (auto k = 0; k < ksize; ++k)
      kernels(n, k + kernel_radius - kradius) /= kernel_sum;
  }

  Eigen::IOFormat HeavyFmt(3, 0, ", ", ",\n", "[", "]", "[", "]");
  SARA_CHECK(Eigen::Map<const Eigen::RowVectorXf>(  //
      sigmas.data(),                                //
      sigmas.size())                                //
  );
  SARA_CHECK(kernels.sizes().reverse().transpose());
  SARA_DEBUG << "stacked kernels =\n"
             << kernels.matrix().transpose().format(HeavyFmt) << std::endl;


  SARA_DEBUG << "Copying the stacked kernels to CUDA constant memory"
             << std::endl;
  shakti::tic();
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(constant_gauss_kernels,  //
                                           kernels.data(),
                                           kernels.size() * sizeof(float)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(constant_gauss_kernel_sizes,
                                           kernel_sizes.data(),
                                           kernel_sizes.size() * sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(constant_kernel_count,  //
                                           kernels.sizes().data(),
                                           sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(constant_kernel_size,  //
                                           kernels.sizes().data() + 1,
                                           sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(constant_gauss_kernel_radius,  //
                                           &kernel_radius, sizeof(int)));
  shakti::toc("copy to constant memory");

#define THIS_WORKS
#ifdef THIS_WORKS
  auto kernels_copied = sara::Tensor_<float, 2>{kernels.sizes()};
  kernels_copied.flat_array().fill(-1);
  SARA_DEBUG << "kernels copied (initialized)=\n"
             << kernels_copied.matrix().transpose().format(HeavyFmt)
             << std::endl;

  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(kernels_copied.data(),
                                             constant_gauss_kernels,
                                             kernels.size() * sizeof(float)));
  SARA_DEBUG << "kernels copied=\n"
             << kernels_copied.matrix().transpose().format(HeavyFmt)
             << std::endl;

  auto kernel_size = int{};
  auto kernel_count = int{};
  auto kernel_radius_point_copied = int{};
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_size,  //
                                             constant_kernel_size,
                                             sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_count,  //
                                             constant_kernel_count,
                                             sizeof(int)));
  SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(&kernel_radius_point_copied,  //
                                             constant_gauss_kernel_radius,
                                             sizeof(int)));

  SARA_CHECK(kernel_size);
  SARA_CHECK(kernel_count);
  SARA_CHECK(kernel_radius_point_copied);
#endif

  const auto w = 4 * 1920;
  const auto h = 4 * 1080;

  // Initialize the octave CUDA surface.
  auto octave = sc::make_gaussian_octave<float>(w, h, scale_count);
  octave.init_surface();
  BOOST_CHECK_EQUAL(octave.scale_count(), scale_count + 3);

  // Initialize the octave.
  auto dirac = sara::Image<float, 3>{w, h, octave.scale_count()};
  dirac.flat_array().fill(0);
  for (auto s = 0; s < dirac.depth(); ++s)
    dirac(w / 2, h / 2, s) = 1;
  copy(dirac, octave);

  // Convolve the octave.
  auto conv_x = sc::make_gaussian_octave<float>(w, h, scale_count);
  auto conv_y = sc::make_gaussian_octave<float>(w, h, scale_count);
  conv_x.init_surface();
  conv_y.init_surface();

  shakti::tic();
  {
    const auto threadsperBlock = dim3(tile_x, tile_y, tile_z);
    const auto numBlocks = dim3(
        (octave.width() + threadsperBlock.x - 1) / threadsperBlock.x,
        (octave.height() + threadsperBlock.y - 1) / threadsperBlock.y,
        (octave.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);

    // x-convolution.
    convolve_x<<<numBlocks, threadsperBlock>>>(                //
        octave.surface_object(),                               //
        conv_x.surface_object(),                               //
        /* input_layer */ 0,                                   //
        octave.width(), octave.height(), octave.scale_count()  //
    );
  }

  // y-convolution.
  {
    const auto threadsperBlock = dim3(tile_x, tile_y, tile_z);
    const auto numBlocks = dim3(
        (octave.width() + threadsperBlock.x - 1) / threadsperBlock.x,
        (octave.height() + threadsperBlock.y - 1) / threadsperBlock.y,
        (octave.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);

    convolve_y<<<numBlocks, threadsperBlock>>>(                //
        conv_x.surface_object(),                               //
        conv_y.surface_object(),                               //
        octave.width(), octave.height(), octave.scale_count()  //
    );
  }
  shakti::toc("Gaussian convolution");


  auto values = dirac;
  copy(conv_y, values);

  if (w < 10 && h < 10)
  {
    for (auto s = 0; s < values.depth(); ++s)
    {
      SARA_CHECK(s);
      std::cout << sara::tensor_view(values)[s].matrix().format(HeavyFmt)
                << std::endl;
    }
  }
}
