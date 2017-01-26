// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
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
    @ingroup Core
    @defgroup Image Image
    @{
   */

  //! @{
  //! @brief Forward declaration of the image classes.
  template <typename PixelType, int N = 2> class ImageView;

  template <typename PixelType, int N = 2,
            template <typename> class Allocator = std::allocator>
  using Image = MultiArrayBase<ImageView<PixelType, N>, Allocator>;
  //! @}


  //! @brief Forward declaration of the generic color conversion function.
  template <typename SrcImageView, typename DstImageView>
  void convert(const SrcImageView& src, DstImageView& dst);


  //! @brief The image base class.
  template <typename PixelType, int N>
  class ImageView : public MultiArrayView<PixelType, N, ColMajor>
  {
    using base_type = MultiArrayView<PixelType, N, ColMajor>;
    using self_type = ImageView;

  public:
    using value_type = PixelType;
    using pixel_type = PixelType;
    using pointer = typename base_type::pointer;
    using vector_type = typename base_type::vector_type;
    using base_type::Dimension;

    //! @{
    //! Matrix views for linear algebra.
    using const_matrix_view_type = Map<
      const Matrix<
        typename ElementTraits<pixel_type>::value_type,
        Dynamic, Dynamic, RowMajor
      >
    >;

    using matrix_view_type = Map<
      Matrix<
        typename ElementTraits<pixel_type>::value_type,
        Dynamic, Dynamic, RowMajor
      >
    >;
    //! @}

  public:
    //! Default image constructor.
    inline ImageView()
      : base_type{}
    {
    }

    //! Image constructor.
    inline ImageView(pointer data, const vector_type& sizes)
      : base_type{ data, sizes }
    {
    }

    //! @{
    //! Image constructors with specified sizes.
    inline explicit ImageView(const vector_type& sizes)
      : base_type{ sizes }
    {
    }

    inline ImageView(int width, int height)
      : base_type{ width, height }
    {
    }

    inline ImageView(int width, int height, int depth)
      : base_type{ width, height, depth }
    {
    }
    //! @}

    //! Return image width.
    inline int width() const
    {
      return base_type::rows();
    }

    //! Return image height.
    inline int height() const
    {
      return base_type::cols();
    }

    //! @{
    //! Return matrix view for linear algebra with Eigen libraries.
    inline matrix_view_type matrix()
    {
      static_assert(Dimension == 2, "MultiArray must be 2D");
      return matrix_view_type{
        reinterpret_cast<typename ElementTraits<pixel_type>::pointer>(
            base_type::data()),
        height(),
        width()
      };
    }

    inline const_matrix_view_type matrix() const
    {
      static_assert(Dimension == 2, "MultiArray must be 2D");
      return const_matrix_view_type{
        reinterpret_cast<typename ElementTraits<pixel_type>::const_pointer>(
            base_type::data()),
        height(),
        width()
      };
    }
    //! @}

    //! @brief Converts image to the specified pixel format.
    template <typename U>
    inline Image<U, N> convert() const
    {
      auto dst = Image<U, Dimension>{ base_type::sizes() };
      DO::Sara::convert(*this, dst);
      return dst;
    }

    //! @{
    //! @brief Perform custom image filter on the image.
    template <typename Filter, typename... Params>
    inline auto compute(const Params&... params) const
        -> Image<typename Filter::template OutPixel<self_type>, N>
    {
      using OutPixel = typename Filter::template OutPixel<self_type>;
      auto dst = Image<OutPixel, N>{ base_type::sizes() };
      Filter{}(*this, dst, params...);
      return dst;
    }

    template <typename Filter, typename... Params>
    inline auto compute(Filter filter, const Params&... params) const
        -> decltype(filter(std::declval<const ImageView>(),
                           std::declval<Params>()...))
    {
      return filter(*this, params...);
    }
    //! @}

    //! @brief Perform coefficient-wise transform.
    template <typename Op>
    inline auto cwise_transform(Op op) const
        -> Image<decltype(op(std::declval<value_type>())), N>
    {
      using Pixel = decltype(op(std::declval<value_type>()));

      auto dst = Image<Pixel, N>{ this->sizes() };

      auto src_pixel = this->begin();
      auto dst_pixel = dst.begin();
      for ( ; src_pixel != this->end(); ++src_pixel, ++dst_pixel)
        *dst_pixel = op(*src_pixel);

      return dst;
    }
  };


  //! @}


} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_IMAGE_HPP */
