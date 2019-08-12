#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <drafts/OpenCL/Core/Device.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_device)
{
  Platform platform = get_platforms().front();
  BOOST_CHECK(!get_devices(platform).empty());
}
