// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Core.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

#define TEST_FLOAT_ALGOS_ONLY

using namespace std;

BEGIN_NAMESPACE_DO

const int nIter = 10;

template <typename T>
void viewWithoutConversion(const Image<T>& I, 
                           const std::string& windowTitle = "DO++")
{
	// Original image.
	Window win = openWindow(I.width(), I.height(), windowTitle);
	displayThreeChannelColorImageAsIs(I);
	click();
	closeWindow(win);
}

#ifndef TEST_FLOAT_ALGOS_ONLY
template <typename Color>
void testImageConversion(const Image<Rgb8>& I)
{
	std::cout << "// ============================================ //" << std::endl;
	Image<Color> Ic;
	convert(Ic, I);
	Color min, max;
	findMinMax(min, max, Ic);

	typedef Matrix<double, ColorTraits<Color>::NumChannels, 1> Vector;
	typedef Matrix<double, 1, ColorTraits<Color>::NumChannels> RowVector;
	Vector m(min.template cast<double>());
	Vector M(max.template cast<double>());

	std::cout << "Min = " << Map<RowVector>(m.data()) << std::endl;
	std::cout << "Max = " << Map<RowVector>(M.data()) << std::endl;
	std::cout << std::endl << std::endl;

	std::cout << "Viewing without doing RGB conversion" << std::endl;
	viewWithoutConversion(Ic, "Viewing without doing RGB conversion");
	std::cout << "Viewing with RGB conversion" << std::endl;
	view(Ic, "Viewing with RGB conversion");
}

#define GRAY_TEST_CONVERSION(Color) \
void testImageConversion_##Color(const Image<Rgb8>& rgb8image)	\
{ \
	std::cout << "// ========================================= //" << std::endl; \
	std::cout << #Color << " conversion check" << std::endl; \
	Image<Color> I; \
	convert(I, rgb8image); \
	Color min, max; \
	findMinMax(min, max, I);\
	\
	typedef Matrix<double, ColorTraits<Color>::NumChannels, 1> Vector;\
	\
	std::cout << "Min = " << double(min) << std::endl;\
	std::cout << "Max = " << double(max) << std::endl;\
	std::cout << std::endl << std::endl;\
	\
	std::cout << "Viewing with RGB conversion" << std::endl;\
	view(I);\
}

GRAY_TEST_CONVERSION(gray8)
GRAY_TEST_CONVERSION(gray8s)
GRAY_TEST_CONVERSION(gray16)
GRAY_TEST_CONVERSION(gray16s)
GRAY_TEST_CONVERSION(gray32)
GRAY_TEST_CONVERSION(gray32s)
GRAY_TEST_CONVERSION(gray32f)
GRAY_TEST_CONVERSION(gray64f)

void testAllImageConversions(const Image<Rgb8>& I)
{
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Original image check" << std::endl;
	Rgb8 min, max;
	findMinMax(min, max, I);

	typedef Matrix<double, 1, 3> RowVector;
	Vector3d m(min.cast<double>());
	Vector3d M(max.cast<double>());

	std::cout << "Min = " << Map<RowVector>(m.data()) << std::endl;
	std::cout << "Max = " << Map<RowVector>(M.data()) << std::endl;
	std::cout << std::endl << std::endl;
	
#define TEST_IMAGE_CONVERSION(Color) \
	std::cout << "// ========================================= //" << std::endl; \
	std::cout << #Color << " conversion check" << std::endl; \
	testImageConversion<Color>(I);

	TEST_IMAGE_CONVERSION(Rgb8)
	TEST_IMAGE_CONVERSION(Rgb8s)
	TEST_IMAGE_CONVERSION(Rgb16)
	TEST_IMAGE_CONVERSION(Rgb16s)
	TEST_IMAGE_CONVERSION(Rgb32)
	TEST_IMAGE_CONVERSION(Rgb32s)
	TEST_IMAGE_CONVERSION(Rgb32f)
	TEST_IMAGE_CONVERSION(Rgb64f)

	TEST_IMAGE_CONVERSION(Yuv8)
	TEST_IMAGE_CONVERSION(Yuv8s)
	TEST_IMAGE_CONVERSION(Yuv16)
	TEST_IMAGE_CONVERSION(Yuv16s)
	TEST_IMAGE_CONVERSION(Yuv32)
	TEST_IMAGE_CONVERSION(Yuv32s)
	TEST_IMAGE_CONVERSION(Yuv32f)
	TEST_IMAGE_CONVERSION(Yuv64f)
#undef TEST_IMAGE_CONVERSION

	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Gray conversion check" << std::endl;
	testImageConversion_gray8(I);
	testImageConversion_gray8s(I);
	testImageConversion_gray16(I);
	testImageConversion_gray16s(I);
	testImageConversion_gray32(I);
	testImageConversion_gray32s(I);
	testImageConversion_gray32f(I);
	testImageConversion_gray64f(I);
}
#endif

template <typename Color>
void testScaling(const Image<Rgb8>& rgb8image)
{
  // View original image.
  Image<Color> I;
  convert(I, rgb8image);
  view(I);

  // Image size rescaling.
  view(scaleDown(I, 2), "Downscaled image");
  view(scaleUp(I, 2), "Upscaled image");

  // Image rescaling with interpolation.
  view(enlarge(I, 2), "Enlarged image");
  view(reduce(I, 2), "Reduced image");
}

void testAllScaling(const Image<Rgb8>& rgb8image)
{
#define TEST_SCALING(Color) \
  std::cout << "// ========================================= //" << std::endl; \
  std::cout << #Color << " rescaling check" << std::endl; \
  testScaling<Color>(rgb8image);

  TEST_SCALING(float);
#ifndef TEST_FLOAT_ALGOS_ONLY
  TEST_SCALING(uchar);
  TEST_SCALING(ushort);
  TEST_SCALING(uint);
  TEST_SCALING(char);
  TEST_SCALING(short);
  TEST_SCALING(int);
  
  TEST_SCALING(double);

  TEST_SCALING(Rgb8s);
  TEST_SCALING(Rgb16s);
  TEST_SCALING(Rgb32s);
  TEST_SCALING(Rgb8);
  TEST_SCALING(Rgb16);
  TEST_SCALING(Rgb32);
  TEST_SCALING(Rgb32f);
  TEST_SCALING(Rgb64f);
#endif
#undef TEST_SCALING
}

template <typename Color>
void testLinearFiltering(const Image<Rgb8>& rgb8image)
{
	// View original image.
	Image<Color> I;
	convert(I, rgb8image);
	view(I);

  // Start timing from now on.
  Timer t;
  double elapsed;
  typedef typename ColorTraits<Color>::ChannelType S;
  S sigma = S(10);
  Image<Color> I2(I);

	// x-gradient
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyRowDerivative(I2, I);
	elapsed = t.elapsed();
	std::cout << "x gradient: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(rowDerivative(I)), "Row derivative");

	// y-gradient
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyColumnDerivative(I2, I);
	elapsed = t.elapsed();
	std::cout << "y gradient: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(columnDerivative(I)), "Column derivative");

	// Laplacian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyLaplacianFilter(I2, I);
	elapsed = t.elapsed();
	std::cout << "Laplacian filter: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(laplacian(I)), "Laplacian");

	// Sobel
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applySobelFilter(I2, I);
	elapsed = t.elapsed();
	std::cout << "Sobel filter: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(sobel(I)), "Sobel");

	// Scharr
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyScharrFilter(I2, I);
	elapsed = t.elapsed();
	std::cout << "Scharr filter: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(scharr(I)), "Scharr");

	// Gaussian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyGaussianFilter(I2, I, sigma);
	elapsed = t.elapsed();
	std::cout << "Gaussian filter: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(gaussian(I, sigma)), "Gaussian");

	// Deriche-Gaussian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		inPlaceDericheBlur(I2, sigma);
	elapsed = t.elapsed();
	std::cout << "Deriche-Gaussian blur: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(I2), "Deriche-Gaussian blur");

	// Prewitt, Roberts Cross, Kirch, Robinson
	view(colorRescale(prewitt(I)), "Prewitt");
	view(colorRescale(robertsCross(I)), "Roberts Cross");
	view(colorRescale(kirsch(I)), "Kirsch");
	view(colorRescale(robinson(I)), "Robinson");
}

void testAllLinearFiltering(const Image<Rgb8>& rgb8image)
{
  // It does not make really make sense to apply algorithms on images 
  // with integral channel types...
	testLinearFiltering<float>(rgb8image);
#ifndef TEST_FLOAT_ALGOS_ONLY
	testLinearFiltering<double>(rgb8image);
	testLinearFiltering<Rgb32f>(rgb8image);
	testLinearFiltering<Rgb64f>(rgb8image);
#endif
}

template <typename T>
void testDiffAlgos(const Image<Rgb8>& rgb8image)
{
	// View original image.
	Image<T> I;
	convert(I, rgb8image);
	view(I);

	// Start timing from now on.
	Image<Matrix<T,2,1> > g(I.sizes());
	Image<T> gn(I.sizes());
	Image<T> go(I.sizes());
	Image<T> lap(I.sizes());
	Timer t;
	double elapsed;

	// Gradient.
	t.restart();
	for (int i = 0; i < nIter; ++i)
		grad(g, I);
	elapsed = t.elapsed();
	std::cout << "gradient computation: elapsed time = " << elapsed << "s" << std::endl;

	// Gradient squared norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		squaredNorm(gn, g);
	elapsed = t.elapsed();
	std::cout << "gradient squared norm computation: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(gn), "Gradient norm");

	// Gradient blue norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		blueNorm(gn, g);
	elapsed = t.elapsed();
	std::cout << "gradient blue norm computation: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(gn), "Gradient norm");

	// Gradient stable norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		stableNorm(gn, g);
	elapsed = t.elapsed();
	std::cout << "gradient stable norm computation: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(gn), "Gradient norm");

	// Gradient orientation
	t.restart();
	for (int i = 0; i < nIter; ++i)
		orientation(go, g);
	elapsed = t.elapsed();
	std::cout << "gradient orientation computation: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(go), "Gradient orientation");
	
	// Laplacian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		del2(lap, I);
	elapsed = t.elapsed();
	std::cout << "Laplacian computation: elapsed time = " << elapsed << "s" << std::endl;
	view(colorRescale(lap), "Laplacian");
}

void testAllDiffAlgos(const Image<Rgb8>& rgb8image)
{
	testDiffAlgos<float>(rgb8image);
	testDiffAlgos<double>(rgb8image);
}

void testScaleSpace(const Image<Rgb8>& rgb8image)
{
	Image<float> I;
	convert(I, rgb8image);

  vector<Image<float> > gaussians;
  vector<ScaleInfo> scaleInfos;
	
	ComputeScaleSpace computeScaleSpace;
  computeScaleSpace(gaussians, scaleInfos, I);
}

END_NAMESPACE_DO

int main()
{
	using namespace DO;
	Image<Rgb8> I;
	if (!load(I, srcPath("Hydrangeas.jpg")))
    return -1;
	view(I);

#define TEST_IMAGE_CONVERSION
#ifdef TEST_IMAGE_CONVERSION
	//testAllImageConversions(I);
#endif
  testAllScaling(I);
	testAllLinearFiltering(I);
	testAllDiffAlgos(I);
	testScaleSpace(I);
  click();

	return 0;
}