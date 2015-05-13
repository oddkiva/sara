// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_IMAGEPROCESSING_LINEARFILTERING_HPP
#define DO_SARA_IMAGEPROCESSING_LINEARFILTERING_HPP


#include <vector>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Pixel.hpp>


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

      T sum(PixelTraits<T>::zero());
      for (int j = 0; j < kernel_size; ++j, ++yj, ++kj)
        sum += *yj * *kj;

      *y = sum;
    }
  }

  // ====================================================================== //
  // Linear filters.
  /*!
    \brief Apply 1D filter to each image row.
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

    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(w,h);

    std::vector<T> buffer(w+half_size*2);
    for (int y = 0; y < h; ++y)
    {
      // Copy to work array and add padding.
      for (int x = 0; x < half_size; ++x)
        buffer[x] = src(0,y);
      for (int x = 0; x < w; ++x)
        buffer[half_size+x] = src(x,y);
      for (int x = 0; x < half_size; ++x)
        buffer[w+half_size+x] = src(w-1,y);

      convolve_array(&buffer[0], kernel, w, kernel_size);

      for (int x = 0; x < w; ++x)
        dst(x,y) = buffer[x];
    }
  }

  /*!
    \brief Apply 1D filter to each image column.
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
    const int half_size = kernel_size/2;

    // Resize if necessary.
    if (dst.sizes() != src.sizes())
      dst.resize(w,h);

    std::vector<T> buffer(h+half_size*2);
    for (int x = 0; x < w; ++x)
    {
      for (int y = 0; y < half_size; ++y)
        buffer[y] = src(x,0);
      for (int y = 0; y < h; ++y)
        buffer[half_size+y] = src(x,y);
      for (int y = 0; y < half_size; ++y)
        buffer[h+half_size+y] = src(x,h-1);

      convolve_array(&buffer[0], kernel, h, kernel_size);

      for (int y = 0; y < h; ++y)
        dst(x,y) = buffer[y];
    }
  }

  //! brief Apply row-derivative to image.
  template <typename T>
  void apply_row_derivative(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S diff[] = { S(-1), S(0), S(1) };
    apply_row_based_filter(src, dst, diff, 3);
  }

  //! \brief Apply column-derivative to image.
  template <typename T>
  void apply_column_derivative(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S diff[] = { S(-1), S(0), S(1) };
    apply_column_based_filter(src, dst, diff, 3);
  }

  //! \brief Apply Gaussian smoothing to image.
  template <typename T>
  void apply_gaussian_filter(
    const Image<T>& src, Image<T>& dst,
    typename PixelTraits<T>::channel_type sigma,
    typename PixelTraits<T>::channel_type gauss_truncate =
      typename PixelTraits<T>::channel_type(4))
  {
    DO_SARA_STATIC_ASSERT(
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

    apply_row_based_filter(src, dst, &kernel[0], kernel_size);
    apply_column_based_filter(src, dst, &kernel[0], kernel_size);

    delete[] kernel;
  }

  //! \brief Apply Sobel filter to image.
  template <typename T>
  void apply_sobel_filter(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S mean_kernel[] = { S(1), S(2), S(1) };
    const S diff_kernel[] = { S(1), S(0), S(-1) };

    Image<T> tmp;

    // Column derivative.
    apply_row_based_filter(src, tmp, mean_kernel, 3);
    apply_column_based_filter(tmp, tmp, diff_kernel, 3);
    // Row-derivative.
    apply_row_based_filter(src, dst, diff_kernel, 3);
    apply_column_based_filter(dst, dst, mean_kernel, 3);
    // Squared norm.
    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  //! \brief Apply Scharr filter to image.
  template <typename T>
  void apply_scharr_filter(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S mean_kernel[] = { S( 3), S(10), S(3) };
    const S diff_kernel[] = { S(-1),  S(0), S(1) };

    Image<T> tmp;

    // Column derivative.
    apply_row_based_filter(src, tmp, mean_kernel, 3);
    apply_column_based_filter(tmp, tmp, diff_kernel, 3);
    // Row derivative.
    apply_row_based_filter(src, dst, diff_kernel, 3);
    apply_column_based_filter(dst, dst, mean_kernel, 3);
    // Squared norm.
    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }

  //! \brief Apply Prewitt filter to image.
  template <typename T>
  void apply_prewitt_filter(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S mean_kernel[] = { S( 1), S(1), S(1) };
    const S diff_kernel[] = { S(-1), S(0), S(1) };

    Image<T> tmp;

    // Column derivative.
    apply_row_based_filter(src, tmp, mean_kernel, 3);
    apply_column_based_filter(tmp, tmp, diff_kernel, 3);
    // Row derivative.
    apply_row_based_filter(src, dst, diff_kernel, 3);
    apply_column_based_filter(dst, dst, mean_kernel, 3);
    // Squared norm.
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
    typedef typename Image<T>::vector_type Vector;

    const int half_kw = kernel_width/2;
    const int half_kh = kernel_height/2;
    const int w = src.width();
    const int h = src.height();

    Vector sizes(src.sizes()+Vector(half_kw*2, half_kh*2));
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
        if (x >= w+half_kw && y < half_kh)
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
        T val(PixelTraits<T>::zero());
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
  void apply_laplacian_filter(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S kernel[9] = {
      S(0), S( 1), S(0),
      S(1), S(-4), S(1),
      S(0), S( 1), S(0)
    };
    apply_2d_non_separable_filter(src, dst, kernel, 3, 3);
  }

  //! \brief Apply Roberts-Cross filter.
  template <typename T>
  void apply_roberts_cross_filter(const Image<T>& src, Image<T>& dst)
  {
    typedef typename PixelTraits<T>::channel_type S;
    const S k1[] = {
      S( 1), S( 0),
      S( 0), S(-1)
    };
    const S k2[] = {
      S( 0), S( 1),
      S(-1), S( 0)
    };

    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());
    Image<T> tmp;
    apply_2d_non_separable_filter(src, tmp, k1, 2, 2);
    apply_2d_non_separable_filter(src, dst, k2, 2, 2);
    dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
  }


  // ====================================================================== //
  // Helper functions for linear filtering
  //! brief Apply row-derivative to image.
  template <typename T>
  inline Image<T> row_derivative(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_row_derivative(src, dst);
    return dst;
  }

  //! brief Apply column-derivative to image.
  template <typename T>
  inline Image<T> column_derivative(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_column_derivative(src, dst);
    return dst;
  }

  //! \brief Apply Gaussian smoothing to image.
  template <typename T, typename S>
  inline Image<T> gaussian(const Image<T>& src, S sigma, S gauss_truncate = S(4))
  {
    Image<T> dst(src.sizes());
    apply_gaussian_filter(src, dst, sigma, gauss_truncate);
    return dst;
  }

  //! \brief Apply Sobel filter to image.
  template <typename T>
  inline Image<T> sobel(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_sobel_filter(src, dst);
    return dst;
  }

  //! \brief Apply Scharr filter to image.
  template <typename T>
  inline Image<T> scharr(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_scharr_filter(src, dst);
    return dst;
  }

  //! \brief Apply Prewitt filter to image.
  template <typename T>
  inline Image<T> prewitt(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_prewitt_filter(src, dst);
    return dst;
  }

  //! \brief Apply Roberts-Cross filter to image.
  template <typename T>
  inline Image<T> roberts_cross(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_roberts_cross_filter(src, dst);
    return dst;
  }

  //! \brief Apply Laplacian filter to image (slow).
  template <typename T>
  inline Image<T> laplacian_filter(const Image<T>& src)
  {
    Image<T> dst(src.sizes());
    apply_laplacian_filter(src, dst);
    return dst;
  }


  // ====================================================================== //
  // Helper 2D linear filtering functors
#define CREATE_2D_ONLY_FILTER_FUNCTOR(FilterName, function)           \
  /*! \brief Helper class to use Image<T,N>::compute<FilterName>() */ \
  template <typename T, int N> struct FilterName;                     \
  template <typename T> struct FilterName<T,2>                        \
  {                                                                   \
    typedef Image<T> return_type;                                     \
    inline FilterName(const Image<T>& src) : src_(src) {}             \
    inline return_type operator()() const { return function(src_); }  \
    const Image<T>& src_;                                             \
  }

#define CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM(FilterName, function)  \
  /*! \brief Helper class to use Image<T,N>::compute<FilterName>() */   \
  template <typename T, int N> struct FilterName;                       \
  template <typename T> struct FilterName<T,2>                          \
  {                                                                     \
    typedef Image<T> return_type;                                       \
    typedef typename PixelTraits<T>::channel_type parameter_type;       \
    inline FilterName(const Image<T>& src) : src_(src) {}               \
    inline return_type operator()(const parameter_type& param) const    \
    { return function(src_, param); }                                   \
    const Image<T>& src_;                                               \
  }

  CREATE_2D_ONLY_FILTER_FUNCTOR(Sobel, sobel);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Scharr, scharr);
  CREATE_2D_ONLY_FILTER_FUNCTOR(Prewitt, prewitt);
  CREATE_2D_ONLY_FILTER_FUNCTOR(RobertsCross, roberts_cross);
  CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM(Gaussian, gaussian);

#undef CREATE_2D_ONLY_FILTER_FUNCTOR
#undef CREATE_2D_ONLY_FILTER_FUNCTOR_WITH_PARAM

  //! @}

}


#endif /* DO_SARA_IMAGEPROCESSING_LINEARFILTERING_HPP */