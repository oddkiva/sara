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

#include <gpufilter.h>
#include <timer.h>
#include <solve.h>
#include <cpuground.h>
#include <error.h>
#include <symbol.h>

#include <gputex.cuh>
//#include "alg4.cu"
//#include "alg5.cu"
#include "gpufilter.cu"

//=== IMPLEMENTATION ============================================================

// Main -------------------------------------------------------------------------

int main( int argc, char** argv ) {

    int width = 1024, height = 1024, runtimes = 1;

    if ((argc>1 && argc!=4) ||
        (argc==4 && (sscanf(argv[1], "%d", &width) != 1 ||
                     sscanf(argv[2], "%d", &height) != 1 ||
                     sscanf(argv[3], "%d", &runtimes) != 1))) {
        std::cerr << APPNAME << " Bad arguments!\n";
        std::cout << APPNAME << " Usage: " << argv[0] << " width height\n";
        return 1;
    }

    int ne = width * height; // number of elements

    std::vector< float > cpu_img(ne), gpu_img(ne);

    srand( 1234 );
    for (int i = 0; i < ne; ++i)
        gpu_img[i] = cpu_img[i] = rand() / (float)RAND_MAX;

    float sigma = 16.f;

    float b10, a11, b20, a21, a22;
    gpufilter::weights1( sigma, b10, a11 );
    gpufilter::weights2( sigma, b20, a21, a22 );

    float me = 0.f, mre = 0.f; // maximum error and maximum relative error

    if (runtimes == 1) {
        std::cout << APPNAME << " Size: " << width << " x " << height << " ; "
                  << " Runtimes: " << runtimes << "\n";
        std::cout << APPNAME << " Weights: " << b10 << " " << a11 << " "
                  << b20 << " " << a21 << " " << a22 << "\n";
        std::cout << APPNAME << " --- CPU reference [1] ---\n";

        gpufilter::scoped_timer_stop sts( gpufilter::timers.cpu_add("CPU") );
        gpufilter::gaussian_cpu( &cpu_img[0], width, height, sigma );
        std::cout << APPNAME << " CPU Timing: "
                  << (width*height)/(sts.elapsed()*1024*1024) << " MiP/s\n";

        std::cout << APPNAME << " --- GPU alg5 ---\n";
    }

    gpufilter::alg_setup algs5;
    gpufilter::dvector<float> d_out5;
    gpufilter::dvector<float> d_transp_pybar5, d_transp_ezhat5, d_ptucheck5,
        d_etvtilde5;
    cudaArray *a_in5;

    gpufilter::alg_setup algs4, algs_transp4;
    gpufilter::dvector<float> d_out4, d_transp_out4;
    gpufilter::dvector<float2> d_transp_pybar4, d_transp_ezhat4, d_pubar4, d_evhat4;
    cudaArray *a_in4;

    // gpufilter::prepare_alg5( algs5, d_out5, d_transp_pybar5, d_transp_ezhat5,
    //                          d_ptucheck5, d_etvtilde5, a_in5, &gpu_img[0], width,
    //                          height, b10, a11 );
    // gpufilter::prepare_alg4( algs4, algs_transp4, d_out4, d_transp_out4, d_transp_pybar4,
    //                          d_transp_ezhat4, d_pubar4, d_evhat4, a_in4, &gpu_img[0],
    //                          width, height, b20, a21, a22 );

    {
        gpufilter::scoped_timer_stop sts( gpufilter::timers.gpu_add("GPU") );
        for (int i = 0; i < runtimes; ++i) {
            //gpufilter::alg5( d_out5, d_transp_pybar5, d_transp_ezhat5, d_ptucheck5,
            //                 d_etvtilde5, a_in5, algs5 );
            //d_out5.copy_to(&gpu_img[0], width*height);
            //cudaMemcpyToArray( a_in4, 0, 0, d_out5, width*height*sizeof(float),
            //                   cudaMemcpyDeviceToDevice );
            //gpufilter::alg4( d_out4, d_transp_out4, d_transp_pybar4, d_transp_ezhat4, d_pubar4,
            //                 d_evhat4, a_in4, algs4, algs_transp4 );
            gpufilter::alg5( &gpu_img[0], width, height, b10, a11 );
            gpufilter::alg4( &gpu_img[0], width, height, b20, a21, a22 );
        }
        //gpufilter::gaussian_gpu(&gpu_img[0], width, height, sigma);
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
        d_out4.copy_to( &gpu_img[0], width * height );

        std::cout << APPNAME << " --- Checking computations ---\n";
        gpufilter::check_cpu_reference( &cpu_img[0], &gpu_img[0], ne, me, mre );
        std::cout << APPNAME << " cpu x gpu: me = " << me << " mre = " << mre << "\n";
    }

    return 0;

}
