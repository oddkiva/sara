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

using namespace std;

namespace DO {

const int nIter = 1;

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

template <typename Color>
void testImageConversion(const Image<Rgb8>& I)
{
	cout << "// ============================================ //" << endl;
	Image<Color> Ic;
	convert(Ic, I);
	Color min, max;
	findMinMax(min, max, Ic);

	typedef Matrix<double, ColorTraits<Color>::NumChannels, 1> Vector;
	typedef Matrix<double, 1, ColorTraits<Color>::NumChannels> RowVector;
	Vector m(min.template cast<double>());
	Vector M(max.template cast<double>());

	cout << "Min = " << Map<RowVector>(m.data()) << endl;
	cout << "Max = " << Map<RowVector>(M.data()) << endl;
	cout << endl << endl;

	cout << "Viewing without doing RGB conversion" << endl;
	viewWithoutConversion(Ic, "Viewing without doing RGB conversion");
	cout << "Viewing with RGB conversion" << endl;
	viewImage(Ic, "Viewing with RGB conversion");
}

#define GRAY_TEST_CONVERSION(Color)                                           \
void testImageConversion_##Color(const Image<Rgb8>& rgb8image)                \
{                                                                             \
	cout << "// ========================================= //" << endl;          \
	cout << #Color << " conversion check" << endl;                              \
	Image<Color> I;                                                             \
	convert(I, rgb8image);                                                      \
	Color min, max;                                                             \
	findMinMax(min, max, I);                                                    \
	                                                                            \
	typedef Matrix<double, ColorTraits<Color>::NumChannels, 1> vector_type;     \
	                                                                            \
	cout << "Min = " << double(min) << endl;                                    \
	cout << "Max = " << double(max) << endl;                                    \
	cout << endl << endl;                                                       \
	                                                                            \
	cout << "Viewing with RGB conversion" << endl;                              \
	viewImage(I);                                                               \
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
	cout << "// ========================================= //" << endl;
	cout << "Original image check" << endl;
	Rgb8 min, max;
	findMinMax(min, max, I);

	typedef Matrix<double, 1, 3> RowVector;
	Vector3d m(min.cast<double>());
	Vector3d M(max.cast<double>());

	cout << "Min = " << Map<RowVector>(m.data()) << endl;
	cout << "Max = " << Map<RowVector>(M.data()) << endl;
	cout << endl << endl;
	
#define TEST_IMAGE_CONVERSION(Color) \
	cout << "// ========================================= //" << endl; \
	cout << #Color << " conversion check" << endl; \
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

	cout << "// ========================================= //" << endl;
	cout << "Gray conversion check" << endl;
	testImageConversion_gray8(I);
	testImageConversion_gray8s(I);
	testImageConversion_gray16(I);
	testImageConversion_gray16s(I);
	testImageConversion_gray32(I);
	testImageConversion_gray32s(I);
	testImageConversion_gray32f(I);
	testImageConversion_gray64f(I);
}

template <typename Color>
void testScaling(const Image<Rgb8>& rgb8image)
{
  // View original image.
  Image<Color> I;
  convert(I, rgb8image);
  viewImage(I);

  // Image size rescaling.
  viewImage(downscale(I, 2), "Downscaled image");
  viewImage(upscale(I, 2), "Upscaled image");

  // Image rescaling with interpolation.
  viewImage(enlarge(I, 2), "Enlarged image");
  viewImage(reduce(I, 2), "Reduced image");
}

void testAllScaling(const Image<Rgb8>& rgb8image)
{
#define TEST_SCALING(Color) \
  cout << "// ========================================= //" << endl; \
  cout << #Color << " rescaling check" << endl; \
  testScaling<Color>(rgb8image);

  TEST_SCALING(float);
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
}

template <typename Color>
void testLinearFiltering(const Image<Rgb8>& rgb8image)
{
	// View original image.
	Image<Color> I;
	convert(I, rgb8image);
	viewImage(I);

  // Start timing from now on.
  typedef typename ColorTraits<Color>::ChannelType S;
  S sigma = S(10);
  Image<Color> I2(I);

  // Timing
  HighResTimer t;
  double elapsed;

	// x-gradient
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyRowDerivative(I2, I);
	elapsed = t.elapsedMs();
	cout << "x gradient: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(rowDerivative(I)), "Row derivative");

	// y-gradient
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyColumnDerivative(I2, I);
  elapsed = t.elapsedMs();
  cout << "y gradient: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(columnDerivative(I)), "Column derivative");

	// Laplacian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyLaplacianFilter(I2, I);
	elapsed = t.elapsedMs();
	cout << "Laplacian filter: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(laplacianFilter(I)), "Laplacian");

	// Sobel
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applySobelFilter(I2, I);
	elapsed = t.elapsedMs();
	cout << "Sobel filter: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(sobel(I)), "Sobel");

	// Scharr
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyScharrFilter(I2, I);
	elapsed = t.elapsedMs();
	cout << "Scharr filter: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(scharr(I)), "Scharr");

	// Gaussian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		applyGaussianFilter(I2, I, sigma);
	elapsed = t.elapsedMs();
	cout << "Gaussian filter: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(gaussian(I, sigma)), "Gaussian");

	// Deriche-Gaussian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		inPlaceDericheBlur(I2, sigma);
	elapsed = t.elapsedMs();
	cout << "Deriche-Gaussian blur: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(I2), "Deriche-Gaussian blur");

	// Prewitt, Roberts Cross, Kirch, Robinson
	viewImage(colorRescale(prewitt(I)), "Prewitt");
	viewImage(colorRescale(robertsCross(I)), "Roberts Cross");
	viewImage(colorRescale(kirsch(I)), "Kirsch");
	viewImage(colorRescale(robinson(I)), "Robinson");
}

void testAllLinearFiltering(const Image<Rgb8>& rgb8image)
{
  // It does not make really make sense to apply algorithms on images 
  // with integral channel types...
	testLinearFiltering<float>(rgb8image);
	testLinearFiltering<double>(rgb8image);
	testLinearFiltering<Rgb32f>(rgb8image);
	testLinearFiltering<Rgb64f>(rgb8image);
}

template <typename T>
void testDiffAlgos(const Image<Rgb8>& rgb8image)
{
	// View original image.
	Image<T> I;
	convert(I, rgb8image);
	viewImage(I);

	// Start timing from now on.
	Image<Matrix<T,2,1> > g(I.sizes());
	Image<T> gn(I.sizes());
	Image<T> go(I.sizes());
	Image<T> lap(I.sizes());
	HighResTimer t;
	double elapsed;

	// Gradient.
	t.restart();
	for (int i = 0; i < nIter; ++i)
		gradient(g, I);
	elapsed = t.elapsedMs();
	cout << "gradient computation: elapsed time = " << elapsed << "ms" << endl;

	// Gradient squared norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		squaredNorm(gn, g);
	elapsed = t.elapsedMs();
	cout << "gradient squared norm computation: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(gn), "Gradient norm");

	// Gradient blue norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		blueNorm(gn, g);
	elapsed = t.elapsedMs();
	cout << "gradient blue norm computation: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(gn), "Gradient norm");

	// Gradient stable norm
	t.restart();
	for (int i = 0; i < nIter; ++i)
		stableNorm(gn, g);
	elapsed = t.elapsedMs();
	cout << "grad stable norm computation: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(gn), "Gradient norm");

	// Gradient orientation
	t.restart();
	for (int i = 0; i < nIter; ++i)
		orientation(go, g);
	elapsed = t.elapsedMs();
	cout << "grad ori computation: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(go), "Gradient orientation");
	
	// Laplacian
	t.restart();
	for (int i = 0; i < nIter; ++i)
		laplacian(lap, I);
	elapsed = t.elapsedMs();
	cout << "Laplacian computation: elapsed time = " << elapsed << "ms" << endl;
	viewImage(colorRescale(lap), "Laplacian");
}

void testAllDiffAlgos(const Image<Rgb8>& rgb8image)
{
	testDiffAlgos<float>(rgb8image);
	testDiffAlgos<double>(rgb8image);
}

void testChainedFiltering(const Image<Rgb8>& I)
{
  viewImage( I.convert<float>().
    compute<Gaussian>(1.5f).
    compute<Laplacian>().
    compute<ColorRescale>() );

  viewImage( I.convert<float>().
    compute<DericheBlur>(1.5f).
    compute<Gradient>().
    compute<DericheBlur>(1.5f).
    compute<SquaredNorm>().
    compute<ColorRescale>() );

  viewImage( I.convert<float>().
    compute<Gradient>().
    compute<Orientation>().
    compute<ColorRescale>() );

  viewImage( I.convert<float>().
    compute<Gradient>().
    compute<SecondMomentMatrix>().
    compute<SquaredNorm>().
    compute<ColorRescale>() );

  // TODO: this crashes because of Hessian evaluation on borders.
  /*view( I.convert<float>().
    compute<Hessian>().
    compute<SquaredNorm>().
    compute<ColorRescale>() );*/
}

} /* namespace DO */

int main()
{
	using namespace DO;
	Image<Rgb8> I;
	if (!load(I, srcPath("Hydrangeas.jpg")))
    return -1;
	viewImage(I);

	testAllImageConversions(I);
  testAllScaling(I);
	testAllLinearFiltering(I);
	testAllDiffAlgos(I);
  testChainedFiltering(I);

	return 0;
}
