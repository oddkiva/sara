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

#ifndef DO_SARA_CORE_IMAGE_IMAGE_HPP
#define DO_SARA_CORE_IMAGE_IMAGE_HPP


#include <DO/Sara/Core/Image/ElementTraits.hpp>
#include <DO/Sara/Core/MultiArray.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup Core
    \defgroup Image Image
    @{
   */

  //! @{
  //! \brief Forward declaration of the image classes.
  template <typename PixelType, int N = 2> class Image;
  template <typename PixelType, int N = 2> class ImageView;
  //! @}


  //! \brief Forward declaration of the generic color conversion function.
  template <typename T, typename U, int N>
  void convert(const Image<T, N>& src, Image<U, N>& dst);


  //! \brief The image base class.
  template <typename MultiArrayType>
  class ImageBase : public MultiArrayType
  {
  public:
    using base_type = MultiArrayType;
    using pixel_type = typename base_type::value_type;
    using pointer = typename base_type::pointer;
    using vector_type = typename base_type::vector_type;
    using base_type::Dimension;

    //! @{
    //! Matrix views for linear algebra.
    using const_matrix_view_type = Map<
      const Matrix<typename ElementTraits<pixel_type>::value_type,
      Dynamic, Dynamic, RowMajor>>;
    using matrix_view_type = Map<
      Matrix<typename ElementTraits<pixel_type>::value_type,
      Dynamic, Dynamic, RowMajor>>;
    //! @}

  public:
    //! Default image constructor.
    inline ImageBase()
      : base_type()
    {
    }

    //! Image constructor.
    inline ImageBase(pointer data, const vector_type& sizes)
      : base_type(data, sizes)
    {
    }

    //! @{
    //! Image constructors with specified sizes.
    inline explicit ImageBase(const vector_type& sizes)
      : base_type(sizes)
    {
    }

    inline ImageBase(int width, int height)
      : base_type(width, height)
    {
    }

    inline ImageBase(int width, int height, int depth)
      : base_type(width, height, depth)
    {
    }
    //! @}

    //! Return image width.
    inline int width() const
    {
      return this->base_type::rows();
    }

    //! Return image height.
    inline int height() const
    {
      return this->base_type::cols();
    }

    //! Return image depth.
    inline int depth() const
    {
      return this->base_type::depth();
    }

    //! @{
    //! Return matrix view for linear algebra with Eigen libraries.
    inline matrix_view_type matrix()
    {
      DO_SARA_STATIC_ASSERT(Dimension == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return matrix_view_type(
        reinterpret_cast<
        typename ElementTraits<pixel_type>::pointer>(base_type::data()),
        height(), width() );
    }

    inline const_matrix_view_type matrix() const
    {
      DO_SARA_STATIC_ASSERT(Dimension == 2, MULTIARRAY_MUST_HAVE_TWO_DIMENSIONS);
      return const_matrix_view_type(
        reinterpret_cast<
        typename ElementTraits<pixel_type>::const_pointer>(base_type::data()),
        height(), width() );
    }
    //! @}
  };


  //! \brief The image view class.
  template <typename T, int N>
  class ImageView : public ImageBase<MultiArrayView<T, N, ColMajor>>
  {
    using base_type = ImageBase<MultiArrayView<T, N, ColMajor>>;

  public: /* interface */
    using vector_type = typename base_type::vector_type;

    inline ImageView(T *data, const vector_type& sizes)
      : base_type(data, sizes)
    {
    }
  };


  //! \brief The image class.
  template <typename T, int N>
  class Image : public ImageBase<MultiArray<T, N, ColMajor>>
  {
    using base_type = ImageBase<MultiArray<T, N, ColMajor>>;

  public: /* interface */
    using vector_type = typename base_type::vector_type;

    //! Default constructor.
    inline Image()
      : base_type()
    {
    }

    //! Constructor that takes ownership of data.
    inline explicit Image(T *data, const vector_type& sizes)
      : base_type(data, sizes)
    {
    }

    //! @{
    //! Constructors with specified sizes.
    inline explicit Image(const vector_type& sizes)
      : base_type(sizes)
    {
    }

    inline Image(int width, int height)
      : base_type(width, height)
    {
    }

    inline Image(int width, int height, int depth)
      : base_type(width, height, depth)
    {
    }
    //! @}

    //! Color conversion method.
    template <typename U>
    Image<U, N> convert() const
    {
      Image<U, N> dst(base_type::sizes());
      DO::Sara::convert(*this, dst);
      return dst;
    }

    //! Convenient helper for chaining filters.
    template <template<typename, int> class Filter>
    inline typename Filter<T, N>::return_type compute() const
    {
      return Filter<T, N>(*this)();
    }

    template <template<typename, int> class Filter>
    inline typename Filter<T, N>::return_type
    compute(const typename Filter<T, N>::parameter_type& param) const
    {
      return Filter<T, N>(*this)(param);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_IMAGE_HPP */
