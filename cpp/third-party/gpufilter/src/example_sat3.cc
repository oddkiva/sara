/**
 *  @file example_sat3.cc
 *  @brief Third SAT (Summed-Area Table) example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <timer.h>
#include <dvector.h>

#include <cpuground.h>

#include <gpufilter.h>

#include <sat.cuh>

#define REPEATS 100

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

    const int in_w = 4096, in_h = 4096;

    std::cout << "[sat3] Generating random input image (" << in_w << "x"
              << in_h << ") ... " << std::flush;

    float *in_cpu = new float[in_w*in_h];
    float *in_gpu = new float[in_w*in_h];

    srand(time(0));

    for (int i = 0; i < in_w*in_h; ++i)
        in_gpu[i] = in_cpu[i] = rand() % 256;

    std::cout << "done!\n[sat3] Computing summed-area table in the CPU ... "
              << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add(
                                              "CPU", in_w*in_h, "iP") );

        gpufilter::sat_cpu( in_cpu, in_w, in_h );
    }

    std::cout << "done!\n[sat3] Configuring the GPU to run ... " << std::flush;

    gpufilter::alg_setup algs;
    gpufilter::dvector<float> d_in_gpu, d_ybar, d_vhat, d_ysum;

    gpufilter::prepare_algSAT( algs, d_in_gpu, d_ybar, d_vhat, d_ysum, in_gpu,
                               in_w, in_h );

    gpufilter::dvector<float> d_out_gpu( algs.width, algs.height );

    std::cout << "done!\n[sat3] Computing summed-area table in the GPU ... "
              << std::flush;

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add(
                                              "GPU", in_w*in_h*REPEATS, "iP") );

        for (int i = 0; i < REPEATS; ++i)
            gpufilter::algSAT( d_out_gpu, d_ybar, d_vhat, d_ysum, d_in_gpu,
                               algs );
    }

    std::cout << "done!\n";

    gpufilter::timers.flush();

    std::cout << "[sat3] Copying result back from the GPU ... " << std::flush;

    d_out_gpu.copy_to( in_gpu, algs.width, algs.height, in_w, in_h );

    std::cout << "done!\n[sat3] Checking GPU result against CPU reference\n";

    float me, mre;

    check_reference( in_cpu, in_gpu, in_w*in_h, me, mre );

    std::cout << "[sat3] Maximum relative error: " << mre << "\n";

    delete [] in_cpu;
    delete [] in_gpu;

    return 0;

}
