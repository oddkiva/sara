/**
 *  @file example_r1.cc
 *  @brief First R (Recursive Filtering) example
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
    const float b0 = 1.f, a1 = -1.f;

    std::cout << "[r1] Generating random input image (" << in_w << "x"
              << in_h << ") ... " << std::flush;

    float *in = new float[in_w*in_h];

    srand(time(0));

    for (int i = 0; i < in_w*in_h; ++i)
        in[i] = rand() % 8;

    std::cout << "done!\n";

    print_matrix(in, in_w, in_h, 2);

    std::cout << "[r1] Recursive filter: y_i = b0 * x_i - a1 * y_{i-1}\n"
              << "[r1] Considering causal filter (only forward) on each row\n"
              << "[r1] Feedforward and feedback coefficients are: b0 = "
              << b0 << " ; a1 = " << a1 << "\n"
              << "[r1] This is equivalent to an inclusive multi-scan with the "
              << "plus operator\n[r1] CPU Computing first-order recursive "
              << "filtering with zero-border ... " << std::flush;

    gpufilter::rrfr( in, in_w, in_h, b0, a1, true );

    std::cout << "done!\n[r1] Output matrix " << in_w << " x "
              << in_h << " :\n";

    print_matrix(in, in_w, in_h, 4);

    delete [] in;

    return 0;

}
