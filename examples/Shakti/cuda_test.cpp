#include <DO/Sara/Graphics.hpp>

#include "Utilities/DeviceInfo.hpp"

#include "toy_test_cuda.h"


using namespace DO;
using namespace std;


__device__
void test()
{
}


GRAPHICS_MAIN()
{
  try
  {
    std::vector<Shakti::Device> devices{Shakti::get_devices()};
    cout << devices.back() << endl;
    toy_test_cuda();

    create_window(200, 200);
    get_key();
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}