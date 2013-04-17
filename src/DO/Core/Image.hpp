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

#ifndef DO_CORE_IMAGE_HPP
#define DO_CORE_IMAGE_HPP

namespace DO {

  // ======================================================================== //
  /*!
    \ingroup Core
    \defgroup Image Image
    @{
   */

  //! \brief The specialized element traits class when the entry is a color.
	template <typename T, typename Layout>
	struct ElementTraits<Color<T, Layout> >
	{
		typedef Array<T, Layout::size, 1> value_type; //!< STL-like typedef.
		typedef size_t size_type; //!< STL-like typedef.
		typedef value_type * pointer; //!< STL-like typedef.
		typedef const value_type * const_pointer; //!< STL-like typedef.
		typedef value_type& reference; //!< STL-like typedef.
		typedef const value_type& const_reference; //!< STL-like typedef.
		typedef value_type * iterator; //!< STL-like typedef.
		typedef const value_type * const_iterator; //!< STL-like typedef.
		static const bool is_scalar = false; //!< STL-like typedef.
	};

  //! The forward declaration of the image class.
  template <typename Color, int N = 2> class Image;

  //! \brief Helper function for color conversion.
  template <typename T, typename U, int N>
	void convert(Image<T, N>& dst, const Image<U, N>& src);  

  //! \brief The image class.
  template <typename Color, int N>
	class Image : public MultiArray<Color, N, ColMajor>
	{
		typedef MultiArray<Color, N, ColMajor> Base;

	public: /* interface */
    //! N-dimensional integral vector type.
    typedef typename Base::Vector Vector;
    
    //! Default constructor.
		inline Image()
			: Base() {}

    //! Constructor with specified sizes.
		inline explicit Image(const Vector& sizes)
			: Base(sizes) {}

    //! Constructor which wraps raw data.
		inline Image(Color *data, const Vector& sizes)
			: Base(data, sizes) {}

    //! Constructor with specified sizes.
		inline Image(int width, int height)
			: Base(width, height) {}

    //! Constructor with specified sizes.
    inline Image(int width, int height, int depth)
      : Base(width, height, depth) {}

    //! Copy constructor.
		inline Image(const Base& x)
			: Base(x) {}

    //! Assignment operators.
		inline const Image& operator=(const Image& I)
		{ Base::operator=(I); return *this;}

    //! Constant width accessor.
		inline int width() const { return this->Base::rows(); }

    //! Constant height accessor.
		inline int height() const {	return this->Base::cols(); }

    //! Constant depth accessor (only for volumetric image.)
		inline int depth() const {	return this->Base::depth(); }

    //! Color conversion methods.
    template <typename Color2>
    Image<Color2, N> convert() const
    {
      Image<Color2, N> dst(Base::sizes());
      DO::convert(dst, *this);
      return dst;
    }
	};

	// ====================================================================== //
	// Construct image views from row major multi-array.
  //! \todo Check this functionality...
#define DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY(Colorspace)          \
  /*! \brief Reinterpret column-major matrix as an image. */            \
	template <typename T>                                                 \
	inline Image<Color<T, Colorspace>, Colorspace::size>                  \
  as##Colorspace##Image(const MultiArray<Matrix<T,3,1>,                 \
                                         Colorspace::size,              \
                                         ColMajor>& M)                  \
	{                                                                     \
		return Image<Color<T, Colorspace> >(                                \
			reinterpret_cast<Color<T, Colorspace> *>(M.data()),               \
			M.sizes() );                                                      \
	}

  DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY(Rgb)
  DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY(Rgba)
  DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY(Cmyk)
  DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY(Yuv)
#undef DEFINE_IMAGE_VIEW_FROM_COLMAJOR_MULTIARRAY


	// ====================================================================== //
	// Generic image conversion function.
  //! \brief Generic image converter class.
  template <typename T, typename U, int N>
  struct ConvertImage {
    //! Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<U, N>& src)
    {
      if (dst.sizes() != src.sizes())
        dst.resize(src.sizes());

      const U *src_first = src.data();
      const U *src_last = src_first + src.size();

      T *dst_first  = dst.data();

      for ( ; src_first != src_last; ++src_first, ++dst_first)
        convertColor(*dst_first, *src_first);
    }
  };

  //! \brief Specialized image converter class when the source and color types
  //! are the same.
  template <typename T, int N>
  struct ConvertImage<T,T,N> {
    //! Implementation of the image conversion.
    static void apply(Image<T, N>& dst, const Image<T, N>& src)
    {
      dst = src;
    }
  };

	template <typename T, typename U, int N>
	inline void convert(Image<T, N>& dst, const Image<U, N>& src)
	{
    ConvertImage<T,U,N>::apply(dst, src);
	}


	// ====================================================================== //
	// Find min and max values in images according to point-wise comparison.
  //! \brief Find min and max pixel values of the image.
	template <typename T, int N, typename Layout>
	void findMinMax(Color<T, Layout>& min, Color<T, Layout>& max,
                  const Image<Color<T, Layout>, N>& src)
	{
		const Color<T,Layout> *src_first = src.data();
		const Color<T,Layout> *src_last = src_first + src.size();

		min = *src_first;
		max = *src_first;

		for ( ; src_first != src_last; ++src_first)
		{
			min = min.cwiseMin(*src_first);
			max = max.cwiseMax(*src_first);
		}
	}

  //! Macro that defines min-max value functions for a specific grayscale 
  //! color types.
#define DEFINE_FINDMINMAX_GRAY(T)                               \
  /*! \brief Find min and max grayscale values of the image. */ \
	template <int N>                                              \
	inline void findMinMax(T& min, T& max, const Image<T, N>& src)\
	{                                                             \
		const T *src_first = src.data();                            \
		const T *src_last = src_first + src.size();                 \
                                                                \
		min = *std::min_element(src_first, src_last);               \
		max = *std::max_element(src_first, src_last);               \
	}

	DEFINE_FINDMINMAX_GRAY(uchar)
	DEFINE_FINDMINMAX_GRAY(char)
	DEFINE_FINDMINMAX_GRAY(ushort)
	DEFINE_FINDMINMAX_GRAY(short)
	DEFINE_FINDMINMAX_GRAY(uint)
	DEFINE_FINDMINMAX_GRAY(int)
	DEFINE_FINDMINMAX_GRAY(float)
	DEFINE_FINDMINMAX_GRAY(double)
#undef DEFINE_FINDMINMAX_GRAY


	// ====================================================================== //
	// Image rescaling functions
  //! \brief color rescaling function.
	template <typename T, typename Layout, int N>
	inline Image<Color<T,Layout>, N> colorRescale(
		const Image<Color<T,Layout>, N>& src,
		const Color<T, Layout>& a = black<T>(),
		const Color<T, Layout>& b = white<T>())
	{
		Image<Color<T,Layout>, N> dst(src.sizes());

		const Color<T,Layout> *src_first = src.data();
		const Color<T,Layout> *src_last = src_first + src.size();
		Color<T,Layout> *dst_first  = dst.data();

		Color<T,Layout> min(*src_first);
		Color<T,Layout> max(*src_first);
		for ( ; src_first != src_last; ++src_first)
		{
			min = min.cwiseMin(*src_first);
			max = max.cwiseMax(*src_first);
		}

		if (min == max)
		{
			std::cerr << "Warning: min == max!" << std::endl;
			return dst;
		}

		for (src_first = src.data(); src_first != src_last; 
			 ++src_first, ++dst_first)
			*dst_first = a + (*src_first-min).cwiseProduct(b-a).
                                        cwiseQuotient(max-min);

		return dst;
	}

  //! Macro that defines a color rescaling function for a specific grayscale 
  //! color type.
#define DEFINE_RESCALE_GRAY(T)											          \
  /*! \brief Rescales color values properly for viewing. */   \
	template <int N>													                  \
	inline Image<T, N> colorRescale(const Image<T, N>& src,	    \
									T a = ColorTraits<T>::min(),		            \
									T b = ColorTraits<T>::max())		            \
	{																	                          \
		Image<T, N> dst(src.sizes());									            \
																		                          \
		const T *src_first = src.data();								          \
		const T *src_last = src_first + src.size();						    \
		T *dst_first  = dst.data();										            \
																		                          \
		T min = *std::min_element(src_first, src_last);				    \
		T max = *std::max_element(src_first, src_last);				    \
																		                          \
		if (min == max)													                  \
		{																                          \
      std::cerr << "Warning: min == max!" << std::endl;       \
			return dst;													                    \
		}																                          \
																		                          \
		for ( ; src_first != src_last; ++src_first, ++dst_first)	\
			*dst_first = a + (b-a)*(*src_first-min)/(max-min);			\
                                                              \
		return dst;														                    \
	}

	DEFINE_RESCALE_GRAY(uchar)
	DEFINE_RESCALE_GRAY(char)
	DEFINE_RESCALE_GRAY(ushort)
	DEFINE_RESCALE_GRAY(short)
	DEFINE_RESCALE_GRAY(uint)
	DEFINE_RESCALE_GRAY(int)
	DEFINE_RESCALE_GRAY(float)
	DEFINE_RESCALE_GRAY(double)
#undef DEFINE_RESCALE_GRAY


  //! @}

} /* namespace DO */

#endif /* DO_CORE_IMAGE_HPP */