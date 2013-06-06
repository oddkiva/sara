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

static HighResTimer timer;
double time = 0.0;
void tic()
{
  timer.restart();
}

void toc(ofstream& file)
{
  time = timer.elapsedMs();
  file << "Elapsed time = " << time << " ms" << endl << endl;
}

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

template <typename T, int N, int StorageOrder>
void benchmarkLocator(ofstream& file, const Matrix<int,N,1>& imSizes, int iter)
{
  file << "Timing memory allocation for image" << endl;
  tic();
  typedef MultiArray<T, N, StorageOrder> Img;
  typedef Matrix<int,N,1> Vector;
  Img image(imSizes);
  toc(file);
  
  // Work variables.
  const Img& src = image;
  Img dst(imSizes);
  T T_max( ColorTraits<T>::max() );
  T T_min( ColorTraits<T>::min() );

  // Start benchmarking.
  file << "Assignment by pointer iteration" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
    for (T *data = image.data(); data != image.end(); ++data)
      *data = T_min;
  toc(file);

  file << "Assignment by iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
    for (typename Img::iterator it = image.begin(); it != image.end(); ++it)
      *it = T_min;
  toc(file);

  file << "Assignment by range_iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::range_iterator loc = image.begin_range();
    typename Img::range_iterator loc_end = image.end_range();
    for (; loc != loc_end; ++loc)
      *loc = T_max;
  }
  toc(file);

  file << "Assignment by subrange_iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::subrange_iterator sr = image.begin_subrange(Vector::Zero(), image.sizes());
    typename Img::subrange_iterator sr_end = image.end_subrange();
    for (; sr != sr_end; ++sr)
      *sr = T_max;
  }
  toc(file);

  file << "Assignment by coords_iterator" << endl;
  tic();
  typename Img::coords_iterator it, end;
  for (int i = 0; i < iter; ++i)
    for (it = image.begin_coords(); it != end; ++it)
      image(*it) = T_min;
  toc(file);

  file << "Copy from const_iterator to iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::const_iterator src_it = image.begin();
    typename Img::iterator dst_it = dst.begin();
    for ( ; src_it != image.end(); ++src_it, ++dst_it)
      *dst_it = *src_it;
  }
  toc(file);

  file << "Copy from const_range_iterator to range_iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::const_range_iterator src_loc(src.begin_range());
    typename Img::const_range_iterator src_end(src.end_range());
    typename Img::range_iterator dst_loc = dst.begin_range();
    for (; src_loc != src_end; ++src_loc, ++dst_loc)
      *dst_loc = *src_loc;
  }
  toc(file);

  file << "Copy from const_subrange_iterator to subrange_iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::const_subrange_iterator src_sr(src.begin_subrange(Vector::Zero(), src.sizes()));
    typename Img::const_subrange_iterator src_sr_end(src.end_subrange());
    typename Img::subrange_iterator dst_sr(dst.begin_subrange(Vector::Zero(), src.sizes()));
    for (; src_sr != src_sr_end; ++src_sr, ++dst_sr)
      *dst_sr = *src_sr;
  }
  toc(file);


  file << "Copy with coords_iterator" << endl;
  tic();
  for (int i = 0; i < iter; ++i)
  {
    typename Img::coords_iterator it, end;
    for (int i = 0; i < iter; ++i)
      for (it = image.begin_coords(); it != end; ++it)
        dst(*it) = src(*it);
  }
  toc(file);
}

#define BENCHMARK_LOCATOR(ColorType)                \
{                                                   \
  file << "// ======================== //" << endl; \
  file << "// " << #ColorType << endl;              \
}                                                   \
benchmarkLocator<ColorType, N, StorageOrder>(file, imSizes, iter)

template <int N, int StorageOrder>
void benchmarkLocator_ALL(const string& name, const Matrix<int,N,1>& imSizes,
                          int iter)
{
  ofstream file(stringSrcPath(name).c_str());
  if (!file.is_open())
    return;

//#ifdef WIN32
//  // Timing check (TODO: put this to DO_Core_test and check HighResTimer).
//  file << "// ======================== //" << endl;
//  file << "Check 1000ms sleep time" << endl;
//  tic();
//  Sleep(1000);
//  toc(file);
//#endif

  file << "Benchmarking Locator"<< endl;
#ifdef BENCHMARK_REST
  BENCHMARK_LOCATOR(gray8);
  BENCHMARK_LOCATOR(gray16);
  BENCHMARK_LOCATOR(gray32);
  BENCHMARK_LOCATOR(gray8s);
  BENCHMARK_LOCATOR(gray16s);
  BENCHMARK_LOCATOR(gray32s);
#endif
  BENCHMARK_LOCATOR(gray32f);
  BENCHMARK_LOCATOR(gray64f);

#ifdef BENCHMARK_REST
  BENCHMARK_LOCATOR(Rgb8);
  BENCHMARK_LOCATOR(Rgb16);
  BENCHMARK_LOCATOR(Rgb32);
  BENCHMARK_LOCATOR(Rgb8s);
  BENCHMARK_LOCATOR(Rgb16s);
  BENCHMARK_LOCATOR(Rgb32s);
#endif
  BENCHMARK_LOCATOR(Rgb32f);
  BENCHMARK_LOCATOR(Rgb64f);
#ifdef BENCHMARK_REST
  BENCHMARK_LOCATOR(Yuv8);
  BENCHMARK_LOCATOR(Yuv16);
  BENCHMARK_LOCATOR(Yuv32);
  BENCHMARK_LOCATOR(Yuv8s);
  BENCHMARK_LOCATOR(Yuv16s);
  BENCHMARK_LOCATOR(Yuv32s);
  BENCHMARK_LOCATOR(Yuv32f);
  BENCHMARK_LOCATOR(Yuv64f);

  BENCHMARK_LOCATOR(Cmyk8);
  BENCHMARK_LOCATOR(Cmyk16);
  BENCHMARK_LOCATOR(Cmyk32);
  BENCHMARK_LOCATOR(Cmyk8s);
  BENCHMARK_LOCATOR(Cmyk16s);
  BENCHMARK_LOCATOR(Cmyk32s);
  BENCHMARK_LOCATOR(Cmyk32f);
  BENCHMARK_LOCATOR(Cmyk64f);
  
  BENCHMARK_LOCATOR(Rgba8);
  BENCHMARK_LOCATOR(Rgba16);
  BENCHMARK_LOCATOR(Rgba32);
  BENCHMARK_LOCATOR(Rgba8s);
  BENCHMARK_LOCATOR(Rgba16s);
  BENCHMARK_LOCATOR(Rgba32s);
  BENCHMARK_LOCATOR(Rgba32f);
  BENCHMARK_LOCATOR(Rgba64f);

  BENCHMARK_LOCATOR(Color3b);
  BENCHMARK_LOCATOR(Color3ub);
  BENCHMARK_LOCATOR(Color3i);
  BENCHMARK_LOCATOR(Color3ui);
  BENCHMARK_LOCATOR(Color3f);
  BENCHMARK_LOCATOR(Color3d);

  BENCHMARK_LOCATOR(Color4b);
  BENCHMARK_LOCATOR(Color4ub);
  BENCHMARK_LOCATOR(Color4i);
  BENCHMARK_LOCATOR(Color4ui);
  BENCHMARK_LOCATOR(Color4f);
#endif
  BENCHMARK_LOCATOR(Color4d);


  file.close();
}

} /* namespace DO */

int main()
{
  using namespace DO;

  benchmarkLocator_ALL<2, RowMajor>(
    "bench_array2D_rowmajor.txt", Vector2i(2000, 2000), 1);
  benchmarkLocator_ALL<2, ColMajor>(
    "bench_array2D_colmajor.txt", Vector2i(2000, 2000), 1);
  benchmarkLocator_ALL<3, RowMajor>(
    "bench_array3D_rowmajor.txt", Vector3i(200, 200, 200), 1);
  benchmarkLocator_ALL<3, ColMajor>(
    "bench_array3D_colmajor.txt", Vector3i(200, 200, 200), 1);
  return 0;
}
