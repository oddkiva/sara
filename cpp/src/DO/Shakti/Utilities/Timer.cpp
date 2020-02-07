#include <cuda_runtime_api.h>

#include <iostream>

#include <DO/Shakti/Utilities/ErrorCheck.hpp>
#include <DO/Shakti/Utilities/Timer.hpp>


using namespace std;


namespace DO { namespace Shakti {

  Timer::Timer()
  {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  Timer::~Timer()
  {
    SHAKTI_SAFE_CUDA_CALL(cudaEventDestroy(_start));
    SHAKTI_SAFE_CUDA_CALL(cudaEventDestroy(_stop));
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

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  static Timer timer;

  void tic()
  {
    timer.restart();
  }

  void toc(const char *what)
  {
    auto time = timer.elapsed_ms();
    cout << "[" << what << "] Elapsed time = " << time << " ms" << endl;
  }

} /* namespace Shakti */
} /* namespace DO */
