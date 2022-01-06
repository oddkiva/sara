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

#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Octave"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>


namespace sara = DO::Sara;
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


__global__ void fill(cudaSurfaceObject_t output,
                     int width, int height, int scale_count)
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

  for (auto s = 0; s < result.depth(); ++s)
    SARA_DEBUG << s << "\n" << sara::tensor_view(result)[s].matrix() << std::endl;
  BOOST_CHECK(result == gt);
}


__global__ void convolve(cudaSurfaceObject_t input, cudaSurfaceObject_t output,
                         int width, int height, int scale_count)
{
  // Calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < width && y < height && z < scale_count)
  {
    auto out = float{};

#pragma unroll
    for (auto k = -3; k <= 3; ++k)
    {
      float val;
      surf2DLayeredread<float>(&val, input, (x + k) * sizeof(float), y, z,
                               cudaBoundaryModeClamp);
      out += val;
    }
    // out /= 7;
    // out = exp(-out);
    // printf("[%2d %2d %2d] out = %f \n", x, y, z, out);

    surf2DLayeredwrite<float>(out, output, x * sizeof(float), y, z);
  }
}

BOOST_AUTO_TEST_CASE(test_convolve)
{
  static constexpr auto w = 11;
  static constexpr auto h = 11;
  static constexpr auto scale_count = 1;

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

#ifdef THIS_IS_CORRECT
  for (auto s = 0; s < dirac.depth(); ++s)
    std::cout << "dirac[" << s << "] =\n"
              << sara::tensor_view(dirac)[s].matrix() << std::endl;

  auto dirac_copy = sara::Image<float, 3>{w, h, octave.scale_count()};
  copy(octave, dirac_copy);
  for (auto s = 0; s < dirac.depth(); ++s)
    std::cout << "dirac_copy[" << s << "] =\n"
              << sara::tensor_view(dirac_copy)[s].matrix() << std::endl;
#endif

  // Convolve the octave.
  auto octave_convolved = sc::make_gaussian_octave<float>(w, h, scale_count);
  octave_convolved.init_surface();

  cudaStream_t stream = 0;
  cudaStreamCreate(&stream);
  {
    const dim3 threadsperBlock(16, 16, 2);
    const dim3 numBlocks(
        (octave.width() + threadsperBlock.x - 1) / threadsperBlock.x,
        (octave.height() + threadsperBlock.y - 1) / threadsperBlock.y,
        (octave.scale_count() + threadsperBlock.z - 1) / threadsperBlock.z);
    convolve<<<numBlocks, threadsperBlock>>>(
        octave.surface_object(),                                       //
        octave_convolved.surface_object(),
        octave.width(), octave.height(), octave.scale_count()  //
    );

    auto values = dirac;
    copy(octave_convolved, values);

    // for (auto s = 0; s < values.depth(); ++s)
    // {
    //   SARA_CHECK(s);
    //   std::cout << sara::tensor_view(values)[s].matrix() << std::endl;
    // }
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}

// // Simple copy kernel
// __global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
//                            cudaSurfaceObject_t outputSurfObj, int width,
//                            int height)
// {
//   // Calculate surface coordinates
//   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x < width && y < height)
//   {
//     uchar4 data;
//     // Read from input surface
//     surf2Dread(&data, inputSurfObj, x * 4, y);
//     // Write to output surface
//     data.x *= 2;
//     data.y *= 2;
//     data.z *= 2;
//     data.w *= 2;
//     surf2Dwrite(data, outputSurfObj, x * 4, y);
//   }
// }
//
// BOOST_AUTO_TEST_CASE(from_manual)
// {
//
//   // Host code
//   const int height = 3;
//   const int width = 2;
//
//   // Allocate and set some host data
//   unsigned char* h_data =
//       (unsigned char*) std::malloc(sizeof(unsigned char) * width * height *
//       4);
//   for (int i = 0; i < height * width * 4; ++i)
//     h_data[i] = i;
//
//   // Allocate CUDA arrays in device memory
//   cudaChannelFormatDesc channelDesc =
//       cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
//   cudaArray_t cuInputArray;
//   cudaMallocArray(&cuInputArray, &channelDesc, width, height,
//                   cudaArraySurfaceLoadStore);
//   cudaArray_t cuOutputArray;
//   cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
//                   cudaArraySurfaceLoadStore);
//
//   // Set pitch of the source (the width in memory in bytes of the 2D array
//   // pointed to by src, including padding), we dont have any padding
//   const size_t spitch = 4 * width * sizeof(unsigned char);
//   // Copy data located at address h_data in host memory to device memory
//   cudaMemcpy2DToArray(cuInputArray, 0, 0, h_data, spitch,
//                       4 * width * sizeof(unsigned char), height,
//                       cudaMemcpyHostToDevice);
//
//   // Specify surface
//   struct cudaResourceDesc resDesc;
//   memset(&resDesc, 0, sizeof(resDesc));
//   resDesc.resType = cudaResourceTypeArray;
//
//   // Create the surface objects
//   resDesc.res.array.array = cuInputArray;
//   cudaSurfaceObject_t inputSurfObj = 0;
//   cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
//   resDesc.res.array.array = cuOutputArray;
//   cudaSurfaceObject_t outputSurfObj = 0;
//   cudaCreateSurfaceObject(&outputSurfObj, &resDesc);
//
//   // Invoke kernel
//   dim3 threadsperBlock(16, 16);
//   dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
//                  (height + threadsperBlock.y - 1) / threadsperBlock.y);
//   copyKernel<<<numBlocks, threadsperBlock>>>(inputSurfObj, outputSurfObj,
//   width,
//                                              height);
//
//   // Copy data from device back to host
//   cudaMemcpy2DFromArray(h_data, spitch, cuOutputArray, 0, 0,
//                         4 * width * sizeof(unsigned char), height,
//                         cudaMemcpyDeviceToHost);
//
//   // Destroy surface objects
//   cudaDestroySurfaceObject(inputSurfObj);
//   cudaDestroySurfaceObject(outputSurfObj);
//
//   // Free device memory
//   cudaFreeArray(cuInputArray);
//   cudaFreeArray(cuOutputArray);
//
//   for (auto i = 0 ; i < height * width; ++i)
//   {
//     std::cout << i << " " << int(h_data[i]) << std::endl;
//   }
//
//   // Free host memory
//   free(h_data);
// }
