/**
 *  @file alg5.cu
 *  @brief Algorithm 5 in the GPU
 *  @author Andre Maximo
 *  @date Sep, 2012
 */

#define APPNAME "[alg5]"

//== INCLUDES ===================================================================

#include <cstdio>
#include <iostream>
#include <vector>

#include <timer.h>
#include <solve.h>
#include <cpuground.h>
#include <error.h>
#include <symbol.h>

#include <gputex.cuh>
#include "alg5.cu"

//=== IMPLEMENTATION ============================================================

// Main -------------------------------------------------------------------------

int main( int argc, char** argv ) {

    int width = 1024, height = 1024, runtimes = 1;

    if ((argc>1 && argc!=4) ||
        (argc==4 && (sscanf(argv[1], "%d", &width) != 1 ||
                     sscanf(argv[2], "%d", &height) != 1 ||
                     sscanf(argv[3], "%d", &runtimes) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cout << APPNAME << " Usage: " << argv[0] << " [width height runtimes]\n";
        return 1;
    }

    int ne = width * height; // number of elements

    std::vector< float > cpu_img(ne), gpu_img(ne);

    srand( 1234 );
    for (int i = 0; i < ne; ++i)
        gpu_img[i] = cpu_img[i] = rand() / (float)RAND_MAX;

    float b0 = spline::w0;
    float a1 = spline::w1;

    float me = 0.f, mre = 0.f; // maximum error and maximum relative error

    if (runtimes == 1) {
        std::cout << APPNAME << " Size: " << width << " x " << height << " ; "
                  << " Runtimes: " << runtimes << "\n";
        std::cout << APPNAME << " Weights: " << b0 << " " << a1 << "\n";
        std::cout << APPNAME << " --- CPU reference [1] ---\n";

        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );
        gpufilter::r( &cpu_img[0], width, height, b0, a1 );
        std::cout << APPNAME << " CPU Timing: "
                  << (width*height)/(sts.elapsed()*1024*1024) << " MiP/s\n";

        std::cout << APPNAME << " --- GPU alg5 ---\n";
    }

    gpufilter::alg_setup algs;
    gpufilter::dvector<float> d_out;
    gpufilter::dvector<float> d_transp_pybar, d_transp_ezhat, d_ptucheck,
        d_etvtilde;
    cudaArray *a_in;

    gpufilter::prepare_alg5( algs, d_out, d_transp_pybar, d_transp_ezhat,
                             d_ptucheck, d_etvtilde, a_in, &gpu_img[0], width,
                             height, b0, a1 );
    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );
        for (int i = 0; i < runtimes; ++i)
            gpufilter::alg5( d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck,
                             d_etvtilde, a_in, algs );
        if (runtimes == 1) {
            std::cout << APPNAME << " GPU Timing: "
                      << (width*height*runtimes)/(double)(
                          sts.elapsed()*1024*1024) << " MiP/s\n";
        } else {
            std::cout << std::fixed << (width*height*runtimes)/(double)(
                sts.elapsed()*1024*1024) << "\n";
        }
    }

    if (runtimes == 1) {
        d_out.copy_to( &gpu_img[0], width * height );

        std::cout << APPNAME << " --- Checking computations ---\n";
        gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], ne, me, mre );
        std::cout << APPNAME << " cpu x gpu: me = " << me << " mre = " << mre << "\n";
    }

    return 0;

}
