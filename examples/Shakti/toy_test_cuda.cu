#include <cuda.h>

#include "MultiArray/Matrix.hpp"
#include "MultiArray/MultiArray.hpp"

#include "ImageProcessing.hpp"

#include "toy_test_cuda.h"


void toy_test_cuda()
{
  using namespace std;

  namespace S = DO::Shakti;
  using Vector3i = S::Vector<int, 3> ;
  using Volume3d = S::MultiArray<double, 3>;

  Volume3d volume;
  volume.resize(Vector3i(2, 3, 4));

  dim3 blocks(1, 1, 1);
  dim3 threads(2, 3, 4);

  init<double, 3><<<blocks, threads>>>(volume.data(), volume.size());

  gradient<<<blocks, threads>>>(volume.data(), volume.data(),
                                volume.size(), volume.strides());

  std::vector<double> out;
  volume.to_std_vector(out);

  auto at = [&](int i, int j, int k) {
    return Vector3i(i, j, k).dot(volume.strides());
  };

  for (int i = 0; i < volume.size(0); ++i)
    for (int j = 0; j < volume.size(1); ++j)
      for (int k = 0; k < volume.size(2); ++k)
      {
        Vector3i p(i, j, k);
        cout << "out[" << p << "] = " << out[at(i, j, k)] << endl;
      }
}