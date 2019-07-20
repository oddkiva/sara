#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <drafts/OpenCL/Core/CommandQueue.hpp>
#include <drafts/OpenCL/Core/Context.hpp>
#include <drafts/OpenCL/Core/Device.hpp>
#include <drafts/OpenCL/Core/DeviceBuffer.hpp>
#include <drafts/OpenCL/Core/Kernel.hpp>
#include <drafts/OpenCL/Core/Platform.hpp>
#include <drafts/OpenCL/Core/Program.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_program)
{
  // Write OpenCL program.
  const char *source = R"opencl_code(
    __kernel void square_array(__global float *in_out_array,
                               const unsigned int in_out_array_size)
    {
      int i = get_global_id(0);
      in_out_array[i] *= in_out_array[i];
    }
  )opencl_code";

  // Select platform.
  Platform platform = get_platforms().back();
  // Choose one compute device.
  Device device = get_devices(platform, CL_DEVICE_TYPE_ALL).back();
  std::cout << device << std::endl;
  // Create a context from the selected device.
  Context context(device);
  // Create a command queue for the (context, device) pair.
  CommandQueue queue(context, device);
  // Create an OpenCL program.
  Program program(context, device);
  program.create_from_source(source);
  program.build();
  // Get the only one kernel.
  vector<Kernel> kernels;
  get_kernels_from_program(kernels, program);
  Kernel& kernel = kernels.front();

  // Run the OpenCL program on a sample test.
  auto array = std::vector<float>{1.f, 2.f, 3.f, 4.f};
  auto array_sz = array.size();
  DeviceBuffer<float> array_buffer(context, array.data(), array.size(),
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  // Set the parameter for the following kernel.
  kernel.set_argument(array_buffer, 0);
  kernel.set_argument(array_sz, 1);

  size_t work_items_per_kernel[] = {array.size()};
  queue.enqueue_nd_range_kernel(
    kernel, 1, nullptr, work_items_per_kernel, nullptr);

  // Wait for the commands to get serviced before reading back results.
  queue.finish();

  // Read the results.
  queue.enqueue_read_buffer(array_buffer, array.data());
  BOOST_CHECK_EQUAL(array[0], 1.f);
  BOOST_CHECK_EQUAL(array[1], 4.f);
  BOOST_CHECK_EQUAL(array[2], 9.f);
  BOOST_CHECK_EQUAL(array[3], 16.f);
}
