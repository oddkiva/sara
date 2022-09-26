#include <DO/Sara/Core/DebugUtilities.hpp>

#include <cassert>
#include <stdexcept>


namespace DO::Sara::TensorRT {

  template <typename T>
  __global__ void yolo(const T* conv, T* yolo, int n, int num_boxes,
                       int num_classes)
  {
    const auto i = blockDim.x * blockIdx.x + threadIdx.x;

    yolo[i] = conv[i];
  }

}  // namespace DO::Sara::TensorRT
