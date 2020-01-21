// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <drafts/OpenCL/Core/CommandQueue.hpp>
#include <drafts/OpenCL/Core/DeviceBuffer.hpp>


int main()
{
  using namespace DO::Sara;
  using namespace std;

  // Get the platform.
  cout << "Probing available platforms" << endl;
  auto platforms = get_platforms();
  for (const auto& platform : platforms)
    cout << platform << endl;
  cout << endl;

  // Use the GPU device.
  cout << "Choosing GPU device" << endl;
  auto devices = get_devices(platforms.back(), CL_DEVICE_TYPE_GPU);
  if (devices.empty())
  {
    std::cerr << "No GPU device detected!" << std::endl;
    return EXIT_FAILURE;
  }

  for (const auto& device : devices)
    cout << device << endl;

  // Create a context with the GPU device.
  const auto& gpu_device = devices.front();
  Context context(gpu_device);

  CommandQueue command_queue(context, gpu_device);

  // Build the `square_array` program.
  auto source_filepath = std::string{ src_path("square_array.cl") };
  Program program(context, gpu_device);
  program.create_from_file(source_filepath);
  program.build();

  // Get all the kernels from the program.
  auto kernels = std::vector<Kernel>{};
  if (!get_kernels_from_program(kernels, program))
  {
    std::cerr << "Failed to get kernels from program!" << std::endl;
    return EXIT_FAILURE;
  }
  auto& kernel = kernels.front();

  // Initialize host data.
  constexpr auto N = 4;
  float in_host_vector[N] = {1, 2, 3, 4};
  float out_host_vector[N] = {-1, -1, -1, -1};

  // Initialize device data from host data.
  DeviceBuffer<float> in_device_vector(context, in_host_vector, 4,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
  DeviceBuffer<float> out_device_vector(context, nullptr, 4, CL_MEM_WRITE_ONLY);

  // Pass the arguments to the kernel.
  kernel.set_argument(in_device_vector, 0);
  kernel.set_argument(out_device_vector, 1);
  kernel.set_argument(N, 2);

  // Execute the kernel.
  size_t work_dims = 1;
  size_t work_items_per_kernel[] = {N};
  command_queue.enqueue_nd_range_kernel(
    kernel, 1, nullptr, work_items_per_kernel, nullptr);

  // Wait for the commands to get serviced before reading back results.
  command_queue.finish();

  // Read the results.
  command_queue.enqueue_read_buffer(out_device_vector, out_host_vector);

  for (int i = 0; i < N; ++i)
    cout << out_host_vector[i] << " ";
  cout << endl;

  return EXIT_SUCCESS;
}
