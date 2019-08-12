#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <drafts/OpenCL/Core/Platform.hpp>
#include <drafts/OpenCL/Core/Device.hpp>
#include <drafts/OpenCL/Core/DeviceBuffer.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_buffer)
{
  Platform platform = get_platforms().front();
  Device device = get_devices(platform, CL_DEVICE_TYPE_ALL).front();
  Context context(device);

  float in_array[10];
  DeviceBuffer<float> in_array_buffer(context, in_array, 10,
                                      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
}
