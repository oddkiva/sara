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

//! @file

#ifndef DO_IMAGEPROCESSING_LINEARFILTERING_HPP
#define DO_IMAGEPROCESSING_LINEARFILTERING_HPP


#include <DO/Core/Image.hpp>
#include <DO/Core/Pixel.hpp>


namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup LinearFiltering 2D Linear Filtering
    @{
   */

  /*!
    \brief Convolve a 1D signal \f$f\f$ (or 1D array), with a kernel \f$g\f$.
    @param[in,out]
      signal
      the 1D array containing the 1D signal \f$ f = (f_i)_{1\leq i \leq N}\f$,
      the resulting signal \f$f*g\f$ is stored in signal.
    @param[in]
      kernel
      the convolution kernel \f$g = (g_i)_{1 \leq i \leq K}\f$.
    @param[in] signal_size the signal size \f$N\f$.
    @param[in] kernel_size the kernel size \f$K\f$.
    
   */
  template <typename T>
  void convolve_array(T *signal,
                     const typename PixelTraits<T>::channel_type *kernel,
                     int signal_size, int kernel_size)
  {
    T *yj;
    T *y = signal;
    const typename PixelTraits<T>::channel_type *kj;

    for (int i = 0; i < signal_size; ++i, ++y)
    {
      yj = y;
      kj = kernel;

      T sum(color_min_value<T>());
      for (int j = 0; j < kernel_size; ++j, ++yj, ++kj)
        sum += *yj * *kj;

      *y = sum;
    }
  }

  // ====================================================================== //
  // Linear filters.
  /*!
    \brief Apply 1D filter to image rows. (Slow, deprecated).
    @param[out] dst the row-filtered image.
    @param[in] src the input image
    @param[in] kernel the input kernel
    @param[in] kernel_size the kernel size

    Note that borders are replicated.
   */
  template <typename T>
  void apply_row_based_filter(
    const Image<T>& src, Image<T>& dst,
    const typename PixelTraits<T>::channel_type *kernel,
    int kernel_size)
  {
    const int w = src.width();
    const int h = src.height();
    const int half_size = kernel_size/2;
    Image<T> buffer(w+half_size*2,1);

    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(w,h);


    for (int y = 0; y < h; ++y) {
      // Copy to work array and add padding.
      for (int x = 0; x < half_size; ++x)
        buffer(x,0) = src(0,y);
      for (int x = 0; x < w; ++x)
        buffer(half_size+x,0) = src(x,y);
      for (int x = 0; x < half_size; ++x)
        buffer(w+half_size+x,0) = src(w-1,y);

      // Compute the value by convolution
      for (int x = 0; x < w; ++x) {
        dst(x,y) = color_min_value<T>();
        for (int k = 0; k < kernel_size; ++k)
          dst(x,y) += kernel[k]*buffer(x+k,0);
      }
    }
  }

  /*!
    \brief Apply 1D filter to image columns.
    @param[out] dst the column-filtered image.
    @param[in] src the input image
    @param[in] kernel the input kernel
    @param[in] kernel_size the kernel size
    
    Note that borders are replicated.
   */
  template <typename T>
  void apply_column_based_filter(
    const Image<T>& src, Image<T>& dst,
    const typename PixelTraits<T>::channel_type *kernel,
    int kernel_size)
  {
    const int w = src.width();
    const int h = src.height();
    const int halfSize = kernel_size/2;
    
    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(w,h);

    Image<T> buffer(h+halfSize*2,1);

    for (int x = 0; x < w; ++x)
    {
      // Copy to work array and add padding.
      for (int y = 0; y < halfSize; ++y)
        buffer(y,0) = src(x,0);
      for (int y = 0; y < h; ++y)
        buffer(y+halfSize,0) = src(x,y);
      for (int y = 0; y < halfSize; ++y)
        buffer(h+halfSize+y,0) = src(x,h-1);

      // Compute the value by convolution
      for (int y = 0; y < h; ++y)
      {
        dst(x,y) = color_min_value<T>();
        for (int k = 0; k < kernel_size; ++k)
          dst(x,y) += kernel[k]*buffer(y+k,0);
      }
    }
  }

  /*!
    \brief Apply 1D filter to image rows.
    @param[out] dst the row-filtered image.
    @param[in] src the input image
    @param[in] kernel the input kernel
    @param[in] kernel_size the kernel size

    Note that borders are replicated.
   */
  template <typename T>
  void apply_fast_row_based_filter(
    Image<T>& dst, const Image<T>& src,
    const typename PixelTraits<T>::channel_type *kernel,
    int kernel_size)
  {
    const int w = src.width();
    const int h = src.height();
    const int half_size = kernel_size/2;
    T *buffer = new T[w+half_size*2];
    for (int y = 0; y < h; ++y)
    {
      // Copy to work array and add padding.
      for (int x = 0; x < half_size; ++x)
        buffer[x] = src(0,y);
      for (int x = 0; x < w; ++x)
        buffer[half_size+x] = src(x,y);
      for (int x = 0; x < half_size; ++x)
        buffer[w+half_size+x] = src(w-1,y);

      convolve_array(buffer, kernel, w, kernel_size);
      for (int x = 0; x < w; ++x)
        dst(x,y) = buffer[x];
    }

    delete[] buffer;
  }

  /*!
    \brief Apply 1D filter to image columns. (Slow, deprecated).
    @param[out] dst the column-filtered image.
    @param[in] src the input image
    @param[in] kernel the input kernel
    @param[in] kernel_size the kernel size

    Note that borders are replicated.
   */
  template <typename T>
  void apply_fast_column_based_filter(
    Image<T>& dst, const Image<T>& src,
    const typename PixelTraits<T>::channel_type *kernel,
    int kernel_size)
  {
    const int w = src.width();
    const int h = src.height();
    const int half_size = kernel_size/2;

    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(w,h);

    T *buffer = new T[h+half_size*2];

    for (int x = 0; x < w; ++x)
    {
      for (int y = 0; y < half_size; ++y)
        buffer[y] = src(x,0);
      for (int y = 0; y < h; ++y)
        buffer[half_size+y] = src(x,y);
      for (int y = 0; y < half_size; ++y)
        buffer[h+half_size+y] = src(x,h-1);

      convolve_array(buffer, kernel, h, kernel_size);
      for (int y = 0; y < h; ++y)
        dst(x,y) = buffer[y];
    }

    delete[] buffer;
  }

  //! brief Apply row-derivative to image.
  template <typename T>
  void apply_row_derivative(const Image<T>& src, const Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S diff[] = { S(-1), S(0), S(1) };
    apply_fast_row_based_filter(dst, src, diff, 3);
  }

  //! \brief Apply column-derivative to image.
  template <typename T>
  void apply_column_derivative(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S diff[] = { S(-1), S(0), S(1) };
    apply_fast_column_based_filter(dst, src, diff, 3);
  }

  //! \brief Apply Gaussian smoothing to image.
  template <typename T>
  void apply_gaussian_filter(
    const Image<T>& src, Image<T>& dst,
    typename PixelTraits<T>::channel_type sigma,
    typename PixelTraits<T>::channel_type gauss_truncate = 
      typename PixelTraits<T>::channel_type(4))
  {
    DO_STATIC_ASSERT(
      !std::numeric_limits<typename PixelTraits<T>::channel_type >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );

    typedef typename PixelTraits<T>::channel_type S;

    // Compute the size of the Gaussian kernel.
    int kernel_size = int(S(2) * gauss_truncate * sigma + S(1));
    // Make sure the Gaussian kernel is at least of size 3 and is of odd size.
    kernel_size = std::max(3, kernel_size);
    if (kernel_size % 2 == 0)
      ++kernel_size;

    // Create the 1D Gaussian kernel.
    S *kernel = new S[kernel_size];
    S sum(0);

    // Compute the value of the Gaussian and the normalizing factor.
    for (int i = 0; i < kernel_size; ++i)
    {
      S x = S(i - kernel_size/2);
      kernel[i] = exp(-x*x/(S(2)*sigma*sigma));
      sum += kernel[i];
    }

    // Normalize the kernel.
    for (int i = 0; i < kernel_size; ++i)
      kernel[i] /= sum;

    apply_fast_row_based_filter(dst, src, &kernel[0], kernel_size);
    apply_fast_column_based_filter(dst, dst, &kernel[0], kernel_size);
        
    delete[] kernel;
  }

  //! \brief Apply Sobel filter to image.
  template <typename T>
  void apply_sobel_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S meanKernel[] = { S(1), S(2), S(1) };
    S diffKernel[] = { S(1), S(0), S(-1) };

    Image<T> tmp(src.sizes());
    apply_fast_row_based_filter(tmp, src, meanKernel, 3);
    apply_fast_column_based_filter(tmp, tmp, diffKernel, 3);
    apply_fast_row_based_filter(dst, src, diffKernel, 3);
    apply_fast_column_based_filter(dst, dst, meanKernel, 3);

    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  //! \brief Apply Scharr filter to image.
  template <typename T>
  void apply_scharr_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S meanKernel[] = { S( 3), S(10), S(3) };
    S diffKernel[] = { S(-1),  S(0), S(1) };

    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp(src.sizes());
    apply_fast_row_based_filter(tmp, src, meanKernel, 3);
    apply_fast_column_based_filter(tmp, tmp, diffKernel, 3);
    apply_fast_row_based_filter(dst, src, diffKernel, 3);
    apply_fast_column_based_filter(dst, dst, meanKernel, 3);

    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  //! \brief Apply Prewitt filter to image.
  template <typename T>
  void apply_prewitt_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S meanKernel[] = { S( 1), S(1), S(1) };
    S diffKernel[] = { S(-1), S(0), S(1) };
    
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp(src.sizes());
    apply_fast_row_based_filter(tmp, src, meanKernel, 3);
    apply_fast_column_based_filter(tmp, tmp, diffKernel, 3);
    apply_fast_row_based_filter(dst, src, diffKernel, 3);
    apply_fast_column_based_filter(dst, dst, meanKernel, 3);
    
    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  // ====================================================================== //
  // Non-separable filter functions.
  //! \brief Apply 2D non separable filter to image.
  template <typename T>
  void apply_2d_non_separable_filter(
    const Image<T>& src, Image<T>& dst,
    const typename PixelTraits<T>::channel_type *kernel,
    int kernel_width, int kernel_height)
  {
    typedef typename Image<T>::coords_type Coords;
    
    const int half_kw = kernel_width/2;
    const int half_kh = kernel_height/2;
    const int w = src.width();
    const int h = src.height();

    Coords sizes(src.sizes()+Coords(half_kw*2, half_kh*2));
    Image<T> work(sizes);
    const int workw = work.width();
    const int workh = work.height();
    
    for (int y = 0; y < workh; ++y) {
      for (int x = 0; x < workw; ++x) {
        // North-West
        if (x < half_kw && y < half_kh)
          work(x,y) = src(0,0);
        // West
        if (x < half_kw && half_kh <= y && y < h+half_kh)
          work(x,y) = src(0,y-half_kh);
        // South-West
        if (x < half_kw && y >= h+half_kh)
          work(x,y) = src(0,h-1);
        // North
        if (half_kw <= x && x < w+half_kw && y < half_kh)
          work(x,y) = src(x-half_kw,0);
        // South
        if (half_kw <= x && x < w+half_kw && y >= h+half_kh)
          work(x,y) = src(x-half_kw,h-1);
        // North-East
        if (x >= w+half_kw && y >= h+half_kh)
          work(x,y) = src(w-1,0);
        // East
        if (x >= w+half_kw && half_kh <= y && y < h+half_kh)
          work(x,y) = src(w-1,y-half_kh); 
        // South-East
        if (x >= w+half_kw && y >= h+half_kh)
          work(x,y) = src(w-1,h-1);
        // Middle
        if (half_kw <= x && x < w+half_kw && half_kh <= y && y < h+half_kh)
          work(x,y) = src(x-half_kw,y-half_kh);
      }
    }

    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

    // Convolve
    for (int y = 0; y < h; ++y)
    {
      for (int x = 0; x < w; ++x)
      {
        T val(color_min_value<T>());
        for (int yy = 0; yy < kernel_height; ++yy)
          for (int xx = 0; xx < kernel_width; ++xx)
            val += work(x+xx, y+yy)
                 * kernel[yy*kernel_width + xx];
        dst(x,y) = val;
      }
    }
  }

  //! \brief Apply Laplacian filter (slow).
  template <typename T>
  void apply_laplacian_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S kernel[9] = {
      S(0), S( 1), S(0),
      S(1), S(-4), S(1),
      S(0), S( 1), S(0)
    };
    apply_2d_non_separable_filter(dst, src, kernel, 3, 3);
  }

  //! \brief Apply Roberts-Cross filter.
  template <typename T>
  void apply_roberts_cross_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    S k1[] = { 
      S( 1), S( 0),
      S( 0), S(-1)
    };
    S k2[] = { 
      S( 0), S( 1),
      S(-1), S( 0)
    };

    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp;
    apply_2d_non_separable_filter(tmp, src, k1, 2, 2);
    apply_2d_non_separable_filter(dst, src, k2, 2, 2);
    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  //! \brief Apply Kirsch filter.
  template <typename T>
  void apply_kirsch_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    DO_STATIC_ASSERT(
      !std::numeric_limits<typename PixelTraits<T>::channel_type >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );
    S h1[9] = {
      S(-3)/S(15), S(-3)/S(15), S( 5)/S(15),
      S(-3)/S(15), S( 0)      , S( 5)/S(15),
      S(-3)/S(15), S(-3)/S(15), S( 5)/S(15)
    };

    S h2[9] = {
      S(-3)/S(15), S(-3)/S(15), S(-3)/S(15),
      S(-3)/S(15), S( 0)      , S(-3)/S(15),
      S( 5)/S(15), S( 5)/S(15), S( 5)/S(15)
    };

    S h3[9] = {
      S(-3)/S(15), S(-3)/S(15), S(-3)/S(15),
      S( 5)/S(15), S( 0)      , S(-3)/S(15),
      S( 5)/S(15), S( 5)/S(15), S(-3)/S(15)
    };

    S h4[9] = {
      S( 5)/S(15), S( 5)/S(15), S(-3)/S(15),
      S( 5)/S(15), S( 0)      , S(-3)/S(15),
      S(-3)/S(15), S(-3)/S(15), S(-3)/S(15)
    };

    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp(src.sizes());
    apply_2d_non_separable_filter(tmp, src, h1, 3, 3);
    dst.array() = tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h2, 3, 3);
    dst.array() += tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h3, 3, 3);
    dst.array() += tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h4, 3, 3);
    dst.array() += tmp.array().abs();
    //dst.array().sqrt();
  }

  //! \brief Apply Robinson filter.
  template <typename T>
  void apply_robinson_filter(Image<T>& dst, const Image<T>& src)
  {
    typedef typename PixelTraits<T>::channel_type S;
    DO_STATIC_ASSERT(
      !std::numeric_limits<typename PixelTraits<T>::channel_type >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );
    S h1[9] = {
      S(-1)/S(5), S( 1)/S(5), S( 1)/S(5),
      S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
      S(-1)/S(5), S( 1)/S(5), S( 1)/S(5)
    };

    S h2[9] = {
      S(-1)/S(5), S(-1)/S(5), S(-1)/S(5),
      S( 1)/S(5), S(-2)/S(5), S( 1)/S(5),
      S( 1)/S(5), S( 1)/S(5), S( 1)/S(5)
    };

    S h3[9] = {
      S( 1)/S(5), S( 1)/S(5), S( 1)/S(5),
      S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
      S(-1)/S(5), S(-1)/S(5), S( 1)/S(5)
    };

    S h4[9] = {
      S(-1)/S(5), S(-1)/S(5), S( 1)/S(5),
      S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
      S( 1)/S(5), S( 1)/S(5), S( 1)/S(5)
    };

    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp(src.sizes());
    apply_2d_non_separable_filter(tmp, src, h1, 3, 3);
    dst.array() = tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h2, 3, 3);
    dst.array() += tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h3, 3, 3);
    dst.array() += tmp.array().abs();
    apply_2d_non_separable_filter(tmp, src, h4, 3, 3);
    dst.array() += tmp.array().abs();
    //dst.array().sqrt();
  }

  // ====================================================================== //
  // Helper functions for linear filtering
  //! brief Apply row-derivative to image.
  template <typename T>
  inline Image<T> row_derivative(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_row_derivative(dst, src);
    return dst;
  }

  //! brief Apply column-derivative to image.
  template <typename T>
  inline Image<T> column_derivative(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    applyColumnDerivative(dst, src);
    return dst;
  }

  //! \brief Apply Gaussian smoothing to image.
  template <typename T, typename S>
  inline Image<T> gaussian(const Image<T>& src, S sigma, S gauss_truncate = S(4))
  {
    Image<T> dst(src.sizes());
    apply_gaussian_filter(dst, src, sigma, gauss_truncate);
    return dst;
  }

  //! \brief Apply Sobel filter to image.
  template <typename T>
  inline Image<T> sobel(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_sobel_filter(dst, src);
    return dst;
  }

  //! \brief Apply Scharr filter to image.
  template <typename T>
  inline Image<T> scharr(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_scharr_filter(dst, src);
    return dst;
  }

  //! \brief Apply Prewitt filter to image.
  template <typename T>
  inline Image<T> prewitt(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_prewitt_filter(dst, src);
    return dst;
  }

  //! \brief Apply Roberts-Cross filter to image.
  template <typename T>
  inline Image<T> roberts_cross(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_roberts_cross_filter(dst, src);
    return dst;
  }

  //! \brief Apply Laplacian filter to image (slow).
  template <typename T>
  inline Image<T> laplacian_filter(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_laplacian_filter(dst, src);
    return dst;
  }

  //! \brief Apply Kirsch filter to image.
  template <typename T>
  inline Image<T> kirsch(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_kirsch_filter(dst, src);
    return dst;
  }

  //! \brief Apply Robinson filter to image.
  template <typename T>
  inline Image<T> robinson(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_robinson_filter(dst, src);
    return dst;
  }  


  // ====================================================================== //
  // Helper 2D linear filtering functors
#define CREATE_2D_ONLY_FILTER_FUNCTOR(FilterName, function)           \
  /*! \brief Helper class to use Image<T,N>::compute<FilterName>() */ \
  template <typename T, int N> struct FilterName;                     \
  template <typename T> struct FilterName<T,2>                        \
  {                                                                   \
    typedef Image<T> ReturnType;                                      \
    inline FilterName(const Image<T>& src) : src_(src) {}             \
    inline ReturnType operator()() const { return function(src_); }   \
    const Image<T>& src_;                                             \
  }

#define CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM(FilterName, function)  \
  /*! \brief Helper class to use Image<T,N>::compute<FilterName>() */   \
  template <typename T, int N> struct FilterName;                       \
  template <typename T> struct FilterName<T,2>                          \
  {                                                                     \
    typedef Image<T> ReturnType;                                        \
    typedef typename PixelTraits<T>::channel_type ParamType;             \
    inline FilterName(const Image<T>& src) : src_(src) {}               \
    inline ReturnType operator()(const ParamType& param) const          \
    { return function(src_, param); }                                   \
    const Image<T>& src_;                                               \
  }

  CREATE_2D_ONLY_FILTER_FUNCTOR(Sobel, sobel);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Scharr, scharr);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Prewitt, prewitt);
  CREATE_2D_ONLY_FILTER_FUNCTOR(RobertsCross, roberts_cross);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Robinson, robinson);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Kirsch, kirsch);
  CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM(Gaussian, gaussian);

#undef CREATE_2D_ONLY_FILTER_FUNCTOR
#undef CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM

  //! @}

}


#endif /* DO_IMAGEPROCESSING_LINEARFILTERING_HPP */
