#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <drafts/OpenCL/Core/CommandQueue.hpp>
#include <drafts/OpenCL/Core/Context.hpp>
#include <drafts/OpenCL/Core/Device.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_context)
{
  Platform platform = get_platforms().front();
  Device device = get_devices(platform, CL_DEVICE_TYPE_ALL).front();
  Context context(device);
  BOOST_CHECK(context);
}
