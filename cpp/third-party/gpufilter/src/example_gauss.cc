/**
 *  @file example_gauss.cc
 *  @brief Gaussian filter example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <gpufilter.h>
#include <cpuground.h>

// Check computation
void check_reference( const float *ref,
                      const float *res,
                      const int& ne,
                      float& me,
                      float& mre ) {
    mre = me = (float)0;
    for (int i = 0; i < ne; i++) {
        float a = (float)(res[i]) - ref[i];
        if( a < (float)0 ) a = -a;
        if( ref[i] != (float)0 ) {
            float r = (ref[i] < (float)0) ? -ref[i] : ref[i];
            float b = a / r;
            mre = b > mre ? b : mre;
        }
        me = a > me ? a : me;
    }
}

// Main
int main(int argc, char *argv[]) {

    const int in_w = 1024, in_h = 1024;
    const float sigma = 16.f;

    std::cout << "[gauss] Generating random input image (" << in_w
              << "x" << in_h << ") ... " << std::flush;

    float *in_cpu = new float[in_w*in_h];
    float *in_gpu = new float[in_w*in_h];

    srand(time(0));

    for (int i = 0; i < in_w*in_h; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[gauss] Applying Gaussian filter with sigma = "
              << sigma << "\n[gauss] Considering zero-border as initial "
              << "condition.\n[gauss] Computing in the CPU ... "
              << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

        gpufilter::gaussian_cpu( in_cpu, in_w, in_h, sigma );

        std::cout << "done!\n[gauss] CPU Timing: " << sts.elapsed()*1000
                  << " ms\n";
    }

    std::cout << "[gauss] Computing in the GPU ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::gaussian_gpu( in_gpu, in_w, in_h, sigma );

        std::cout << "done!\n[gauss] GPU Timing: " << sts.elapsed()*1000
                  << " ms\n";
    }

    std::cout << "[gauss] GPU Timing includes memory transfers from and to "
              << "the CPU\n";

    std::cout << "[gauss] Checking GPU result against CPU reference\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, in_w*in_h, me, mre );

    std::cout << std::scientific;

    std::cout << "[gauss] Maximum error: " << me << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
