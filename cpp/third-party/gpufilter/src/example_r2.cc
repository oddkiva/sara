/**
 *  @file example_r2.cc
 *  @brief Second R (Recursive Filtering) example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <cpuground.h>
#include <gpufilter.h>

#include <util.h>

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
    const float b0 = 0.482305, a1 = -0.517695;

    std::cout << "[r2] Generating random input image (" << in_w << "x"
              << in_h << ") ... " << std::flush;

    float *in_cpu = new float[in_w*in_h];
    float *in_gpu = new float[in_w*in_h];

    srand(time(0));

    for (int i = 0; i < in_w*in_h; ++i)
        in_gpu[i] = in_cpu[i] = rand() / (float)RAND_MAX;

    std::cout << "done!\n[r2] Recursive filter: y_i = b0 * x_i - a1 * "
              << "y_{i-1}\n[r2] Considering forward and reverse on rows "
              << "and columns\n[r2] Feedforward and feedback coefficients "
              << "are: b0 = " << b0 << " ; a1 = " << a1 << "\n"
              << "[r2] CPU Computing first-order recursive filtering with "
              << "zero-border ... " << std::flush;

    std::cout << std::fixed << std::setprecision(2);

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );

        gpufilter::r( in_cpu, in_w, in_h, b0, a1 );

        std::cout << "done!\n[r2] CPU Timing: " << sts.elapsed()*1000
                  << " ms\n";
    }

    std::cout << "[r2] GPU Computing first-order recursive filtering with "
              << "zero-border ... " << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );

        gpufilter::alg5( in_gpu, in_w, in_h, b0, a1 );

        std::cout << "done!\n[r2] GPU Timing: " << sts.elapsed()*1000
                  << " ms\n";
    }

    std::cout << "[r2] GPU Timing includes memory transfers from and to "
              << "the CPU\n"
              << "[r2] Checking GPU result against CPU reference\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, in_w*in_h, me, mre );

    std::cout << std::resetiosflags( std::ios_base::floatfield );

    std::cout << "[r2] Maximum relative error: " << mre
              << " ; Maximum error: " << me << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
