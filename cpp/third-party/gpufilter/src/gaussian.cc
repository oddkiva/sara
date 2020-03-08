#include <cstdlib>
#include <iostream>

#include "gpufilter.h"

int main(int argc, char *argv[]) {
	double sigma = atof(argv[1]);
	double b01, a11;
	gpufilter::weights1(sigma, b01, a11);
	double b02, a12, a22;
	gpufilter::weights2(sigma, b02, a12, a22);
	std::cout << "w1: " << b01 << " " << a11 << "\n";
	std::cout << "w2: " << b02 << " " << a12 << " " << a22 << "\n";
	return 0;
}

