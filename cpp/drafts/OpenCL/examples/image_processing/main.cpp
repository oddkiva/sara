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

#ifdef _WIN32
# include <windows.h>
#endif

#if defined(__APPLE__)
# include <OpenCL/cl_gl.h>
#else
# include <CL/cl_gl.h>
# include <GL/glew.h>
#endif

#include <GLFW/glfw3.h>

#include <drafts/OpenCL/Core/CommandQueue.hpp>
#include <drafts/OpenCL/Core/DeviceImage.hpp>
#include <drafts/OpenCL/GL/PixelBuffer.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>


#ifdef _WIN32
void setup_cl_gl_interoperability(DO::Sara::Context& context,
                                  const DO::Sara::Platform& platform)
{
  (void) context;
  (void) platform;
  context.push_property(CL_GL_CONTEXT_KHR, wglGetCurrentContext());
  context.push_property(CL_WGL_HDC_KHR, wglGetCurrentDC());
  context.push_property(CL_CONTEXT_PLATFORM, platform.id);
}
#endif


int main()
{
  using namespace DO::Sara;
  using namespace std;

  // Initialize the windows manager.
  if (!glfwInit())
  {
    std::cerr << "Error: cannot start GLFW!" << std::endl;
    return EXIT_FAILURE;
  }
  SARA_DEBUG << "Init GLFW OK" << std::endl;

#ifndef __APPLE__
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << format("Error: could not start GLEW: %s",
                        glewGetErrorString(err))
              << std::endl;
    return EXIT_FAILURE;
  }
  SARA_DEBUG << "Init GLEW OK" << std::endl;
#endif

  // Query the list of OpenCL platform.
  auto platforms = get_platforms();

  // Detect GPUs.
  auto devices = get_devices(platforms.back(), CL_DEVICE_TYPE_GPU);
  if (devices.empty())
  {
    std::cerr << "No GPU device detected!" << std::endl;
    return EXIT_FAILURE;
  }
  for (const auto& device : devices)
    cout << device << endl;

  // Create a 2D pixel buffer.
  auto cpu_image =
      imread<float>("/Users/david/Desktop/Datasets/sfm/herzjesu_int/0000.png")
          .compute<Resize>(Vector2i{640, 480});
  SARA_DEBUG << "Read image OK" << std::endl;
  SARA_DEBUG << "cpu_image.sizes() = " << cpu_image.sizes().transpose()
             << std::endl;

  // Create a window.
  auto window = glfwCreateWindow(cpu_image.width(), cpu_image.height(),
                                 "Image Processing", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  SARA_DEBUG << "Open window OK" << std::endl;

  // Create a context with the GPU device.
  const auto& gpu_device = devices.front();
  Context context(gpu_device);

  CommandQueue command_queue{context, gpu_device};

#ifdef __WIN32
  // Setup interoperability between OpenGL and OpenCL.
  setup_cl_gl_interoperability(context, platforms.back());
#endif

  // Create a pixel buffer from the host image.
  GL::PixelBuffer<float> pixel_buffer(cpu_image.width(), cpu_image.height(),
                                  cpu_image.data());
  SARA_DEBUG << "Pixel buffer OK" << std::endl;

  // Flush OpenGL operations before using OpenCL.
  glFinish();

  // Use OpenCL for image processing.
  DeviceImage<float> src_image(context, cpu_image.width(), cpu_image.height(),
                               cpu_image.data(),
                               CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY);

  DeviceImage<float> dst_image(context, cpu_image.width(), cpu_image.height(),
                               nullptr, CL_MEM_WRITE_ONLY);

  // Build the program.
  Program program(context, gpu_device);
  program.create_from_file(src_path("gradient.cl"));
  program.build();

  // Prepare the kernel execution.
  auto kernels = std::vector<Kernel>{};
  get_kernels_from_program(kernels, program);
  Kernel& kernel = kernels.front();
  kernel.set_argument(src_image, 0);
  kernel.set_argument(dst_image, 1);
  SARA_DEBUG << "OK" << std::endl;

  // Execute the kernel.
  size_t work_group_size[] = {size_t(cpu_image.width()),
                              size_t(cpu_image.height())};
  command_queue.enqueue_nd_range_kernel(
    kernel, 2, nullptr, work_group_size, nullptr);

  // Read the results.
  command_queue.enqueue_read_image(dst_image, cpu_image.data());

  // Wait for the commands to get serviced before reading back results.
  command_queue.finish();

  pixel_buffer.unpack(cpu_image.data());
  glFinish();

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    glDrawPixels(cpu_image.width(), cpu_image.height(), GL_LUMINANCE, GL_FLOAT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
