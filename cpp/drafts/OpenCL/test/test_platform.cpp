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


BOOST_AUTO_TEST_CASE(test_get_platforms)
{
  vector<Platform> platforms_list = get_platforms();
  BOOST_CHECK(!platforms_list.empty());

  for (const auto& platform : platforms_list)
  {
    BOOST_CHECK(platform.id != nullptr);
    BOOST_CHECK(!platform.name.empty());
    BOOST_CHECK(!platform.vendor.empty());
    BOOST_CHECK(!platform.version.empty());
    BOOST_CHECK(!platform.profile.empty());
    BOOST_CHECK(!platform.extensions.empty());
  }
}
