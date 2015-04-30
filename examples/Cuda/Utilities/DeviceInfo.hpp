#pragma once

#include <iostream>
#include <vector>
#include <utility>

#ifdef _WIN32
# include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


namespace DO {

  class CudaDevice
  {
  public: /* API. */
    int convert_sm_version_to_cores(int major, int minor) const;

    std::string formatted_string() const;

    std::pair<int, int> version() const;

    void make_current_device();

    void make_current_gl_device();

    void reset();

    friend std::ostream& operator<<(std::ostream& os, const CudaDevice& info);

  public: /* data members. */
    int id;
    int driver_version;
    int runtime_version;
    cudaDeviceProp properties;
  };


  int get_num_cuda_devices();

  std::vector<CudaDevice> get_cuda_devices();

} /* namespace DO */