#include <cuda_runtime_api.h>

#include <iostream>

#include <DO/Shakti/Cuda/Utilities/ErrorCheck.hpp>
#include <DO/Shakti/Cuda/Utilities/Timer.hpp>


using namespace std;


namespace DO::Shakti {

  Timer::Timer()
  {
    SHAKTI_SAFE_CUDA_CALL(cudaEventCreate(&_start));
    SHAKTI_SAFE_CUDA_CALL(cudaEventCreate(&_stop));
  }

  Timer::~Timer()
  {
    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
  }

  void Timer::restart()
  {
    cudaEventRecord(_start);
  }

  float Timer::elapsed_ms()
  {
    float ms;
    SHAKTI_SAFE_CUDA_CALL(cudaEventRecord(_stop));
    SHAKTI_SAFE_CUDA_CALL(cudaEventSynchronize(_stop));
    SHAKTI_SAFE_CUDA_CALL(cudaEventElapsedTime(&ms, _start, _stop));
    return ms;
  }


  void tic(Timer& timer)
  {
    timer.restart();
  }

  void toc(Timer& timer, const char* what)
  {
    auto time = timer.elapsed_ms();
    cout << "[" << what << "] Elapsed time = " << time << " ms" << endl;
  }

}  // namespace DO::Shakti
