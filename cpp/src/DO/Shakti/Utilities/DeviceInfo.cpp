#include <sstream>
#include <stdexcept>
#include <string>

#include <DO/Sara/Core/StringFormat.hpp>

#include "DeviceInfo.hpp"
#include "ErrorCheck.hpp"


namespace DO { namespace Shakti {

  using DO::Sara::format;


  int Device::convert_sm_version_to_cores(int major, int minor) const
  {
    // Defines for GPU Architecture types (using the SM version to determine
    // the number of cores per SM
    typedef struct
    {
      // 0xMm (hexadecimal notation)
      // M = SM Major version and m = SM minor version
      int sm;
      int cores;
    } SMCoresPairs;

    SMCoresPairs gpu_arch_cores_per_sm[] =
    {
      { 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
      { 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
      { 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
      { 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
      { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
      { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
      { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
      { 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
      { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
      { 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
      { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
      { -1, -1 }
    };

    int index = 0;
    while (gpu_arch_cores_per_sm[index].sm != -1)
    {
      if (gpu_arch_cores_per_sm[index].sm == ((major << 4) + minor))
        return gpu_arch_cores_per_sm[index].cores;
      index++;
    }

    // If we don't find the values, we default use the previous one to run
    // properly
    using namespace std;
    cout << format("MapSMtoCores for SM %d.%d is undefined. ", major, minor);
    cout << format("Default to use %d Cores/SM\n",
      gpu_arch_cores_per_sm[index - 1].cores);

    return gpu_arch_cores_per_sm[index - 1].cores;
  }

  std::string Device::formatted_string() const
  {
    std::ostringstream os;
    os << format("Device %d: \"%s\"\n", id, properties.name);

    os << format(
      "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
      driver_version / 1000, (driver_version % 100) / 10,
      runtime_version / 1000, (runtime_version % 100) / 10
      );

    os << format(
      "  CUDA Capability Major/Minor version number:    %d.%d\n",
      properties.major, properties.minor);

    os << format(
      "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
      (float)properties.totalGlobalMem / 1048576.0f,
      (unsigned long long) properties.totalGlobalMem);

    os << format(
      "  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
      properties.multiProcessorCount,
      convert_sm_version_to_cores(properties.major, properties.minor),
      convert_sm_version_to_cores(properties.major, properties.minor)
      * properties.multiProcessorCount);

    os << format(
      "  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
      properties.clockRate * 1e-3f, properties.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    os << format(
      "  Memory Clock rate:                             %.0f Mhz\n",
      properties.memoryClockRate * 1e-3f);

    os << format(
      "  Memory Bus Width:                              %d-bit\n",
      properties.memoryBusWidth);

    if (properties.l2CacheSize)
      os << format(
      "  L2 Cache Size:                                 %d bytes\n",
      properties.l2CacheSize);
#endif

    os << format(
      "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
      properties.maxTexture1D,
      properties.maxTexture2D[0], properties.maxTexture2D[1],
      properties.maxTexture3D[0], properties.maxTexture3D[1],
      properties.maxTexture3D[2]);

    os << format(
      "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
      properties.maxTexture1DLayered[0], properties.maxTexture1DLayered[1]);

    os << format(
      "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
      properties.maxTexture2DLayered[0], properties.maxTexture2DLayered[1],
      properties.maxTexture2DLayered[2]);

    os << format(
      "  Total amount of constant memory:               %lu bytes\n",
      properties.totalConstMem);

    os << format(
      "  Total amount of shared memory per block:       %lu bytes\n",
      properties.sharedMemPerBlock);

    os << format(
      "  Total number of registers available per block: %d\n",
      properties.regsPerBlock);

    os << format(
      "  Warp size:                                     %d\n",
      properties.warpSize);

    os << format(
      "  Maximum number of threads per multiprocessor:  %d\n",
      properties.maxThreadsPerMultiProcessor);

    os << format(
      "  Maximum number of threads per block:           %d\n",
      properties.maxThreadsPerBlock);

    os << format(
      "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
      properties.maxThreadsDim[0],
      properties.maxThreadsDim[1],
      properties.maxThreadsDim[2]);

    os << format("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
      properties.maxGridSize[0],
      properties.maxGridSize[1],
      properties.maxGridSize[2]);

    os << format(
      "  Maximum memory pitch:                          %lu bytes\n",
      properties.memPitch);

    os << format(
      "  Texture alignment:                             %lu bytes\n",
      properties.textureAlignment);

    os << format(
      "  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
      (properties.deviceOverlap ? "Yes" : "No"), properties.asyncEngineCount);

    os << format(
      "  Run time limit on kernels:                     %s\n",
      properties.kernelExecTimeoutEnabled ? "Yes" : "No");

    os << format(
      "  Integrated GPU sharing Host Memory:            %s\n",
      properties.integrated ? "Yes" : "No");

    os << format(
      "  Support host page-locked memory mapping:       %s\n",
      properties.canMapHostMemory ? "Yes" : "No");

    os << format(
      "  Alignment requirement for Surfaces:            %s\n",
      properties.surfaceAlignment ? "Yes" : "No");

    os << format(
      "  Device has ECC support:                        %s\n",
      properties.ECCEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    os << format(
      "  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
      properties.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif

    os << format(
      "  Device supports Unified Addressing (UVA):      %s\n",
      properties.unifiedAddressing ? "Yes" : "No");

    os << format(
      "  Device PCI Bus ID / PCI location ID:           %d / %d\n",
      properties.pciBusID, properties.pciDeviceID);

    const char *sComputeMode[] =
    {
      "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
      "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
      "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
      "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
      "Unknown",
      NULL
    };
    os << format("  Compute Mode:\n");
    os << format("    < %s >\n", sComputeMode[properties.computeMode]);

    return os.str();
  }

  std::pair<int, int> Device::version() const
  {
    return std::make_pair(properties.major, properties.minor);
  }

  void Device::make_current_device()
  {
    SHAKTI_SAFE_CUDA_CALL(cudaSetDevice(id));
  }

  void Device::make_current_gl_device()
  {
    SHAKTI_SAFE_CUDA_CALL(cudaGLSetGLDevice(id));
  }

  void Device::reset()
  {
    SHAKTI_SAFE_CUDA_CALL(cudaDeviceReset());
  }

  int Device::warp_size() const
  {
    return properties.warpSize;
  }

  std::ostream& operator<<(std::ostream& os, const Device& info)
  {
    os << info.formatted_string() << "\n";
    return os;
  }

  int get_num_cuda_devices()
  {
    int num_devices = 0;
    cudaError_t error_id = cudaGetDeviceCount(&num_devices);

    if (error_id != cudaSuccess)
    {
      std::string error_msg = format(
        "cudaGetDeviceCount returned %d\n-> %s\n",
        static_cast<int>(error_id),
        cudaGetErrorString(error_id)
        );
      throw std::runtime_error(error_msg.c_str());
    }

    return num_devices;
  }

  std::vector<Device> get_devices()
  {
    int num_devices = get_num_cuda_devices();
    std::vector<Device> devices(num_devices);

    for (auto& device : devices)
    {
      cudaDriverGetVersion(&device.driver_version);
      cudaRuntimeGetVersion(&device.runtime_version);
      cudaGetDeviceProperties(&device.properties, device.id);
    }
    return devices;
  }
} /* namespace Shakti */
} /* namespace DO */
