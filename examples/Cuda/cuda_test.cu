#include "Utilities/DeviceInfo.hpp"
#include "MultiArray/MultiArray.hpp"


using namespace DO;
using namespace std;


__device__
inline int offset()
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  return x*blockDim.y*gridDim.y*blockDim.z*gridDim.z
       + y*blockDim.z*gridDim.z
       + z;
}


template <typename T, int N>
__global__
void compute(Device::Vector<T, N> *array)
{
  int off = offset();
  array[off] = Device::Vector<T, N>::Ones() * T(off);
}


int main(int arc, char **argv)
{
  try
  {
    std::vector<CudaDevice> devices{get_cuda_devices()};
    cout << devices.back() << endl;

    namespace D = Device;
    using Vector3d = D::Vector<double, 3>;
    using Vector3i = D::Vector<int, 3>;
    using Volume3d = D::MultiArray<Vector3d, 3>;

    Volume3d histogram;
    histogram.resize(Vector3i(2, 3, 4));

    dim3 blocks(1, 1, 1);
    dim3 threads(2, 3, 4);
    compute<<<blocks, threads>>>(histogram.data());

    std::vector<Vector3d> out;
    histogram.to_host_vector(out);

    for (int i = 0; i < out.size(); ++i)
      cout << "out[" << i << "] = " << out[i] << endl;
  }
  catch (std::exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}