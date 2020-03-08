/**
 *  @file example_sat1.cc
 *  @brief First SAT (Summed-Area Table) example
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <ctime>
#include <cstdlib>

#include <iostream>
#include <iomanip>

#include <cpuground.h>

// Print a matrix of values
void print_matrix( const float *img,
                   const int& w,
                   const int& h,
                   const int& fw = 4 ) {
    for (int i = 0; i < h; ++i) {
        std::cout << std::setw(fw) << img[i*w];
        for (int j = 1; j < w; ++j)
            std::cout << " " << std::setw(fw) << img[i*w+j];
        std::cout << "\n";
    }
}

// Main
int main(int argc, char *argv[]) {

    const int in_w = 8, in_h = 8;

    std::cout << "[sat1] Generating random input image (" << in_w << "x"
              << in_h << ") ... " << std::flush;

    float *in = new float[in_w*in_h];

    srand(time(0));

    for (int i = 0; i < in_w*in_h; ++i)
        in[i] = rand() % 8;

    std::cout << "done!\n";

    print_matrix(in, in_w, in_h, 2);

    std::cout << "[sat1] Computing summed-area table in the CPU ... "
              << std::flush;

    gpufilter::sat_cpu( in, in_w, in_h );

    std::cout << "done!\n[sat1] Output matrix " << in_w << " x "
              << in_h << " :\n";

    print_matrix(in, in_w, in_h, 4);

    delete [] in;

    return 0;

}
