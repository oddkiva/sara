#define BOOST_TEST_MODULE "OpenCL/Core Utilities"

#include <boost/test/unit_test.hpp>

#include <drafts/OpenCL/Core/CommandQueue.hpp>
#include <drafts/OpenCL/Core/Context.hpp>
#include <drafts/OpenCL/Core/Kernel.hpp>
#include <drafts/OpenCL/Core/Platform.hpp>
#include <drafts/OpenCL/Core/Program.hpp>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_CASE(test_program)
{
  Platform platform = get_platforms().front();
  Device device = get_devices(platform, CL_DEVICE_TYPE_ALL).front();
  Context context(device);
  Program program(context, device);

  const string source = R"opencl_code(
    __kernel void hello_world()
    {
    }
  )opencl_code";

  program.create_from_source(source);
  BOOST_CHECK(program != nullptr);

  program.build();
  cout << program.get_build_logs() << endl;
}