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
  template <typename PixelType, int N = 2> class Image;

  template <typename PixelType, int N = 2> class ImageView;
  //! @}


  //! @brief Forward declaration of the generic color conversion function.
  template <typename SrcImageBase, typename DstImageBase>
  void convert(const SrcImageBase& src, DstImageBase& dst);


  //! @brief The image base class.
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
    inline ImageBase()
      : base_type{}
    {
    }

    //! Image constructor.
    inline ImageBase(pointer data, const vector_type& sizes)
      : base_type{ data, sizes }
    {
    }

    //! @{
    //! Image constructors with specified sizes.
    inline explicit ImageBase(const vector_type& sizes)
      : base_type{ sizes }
    {
    }

    inline ImageBase(int width, int height)
      : base_type{ width, height }
    {
    }

    inline ImageBase(int width, int height, int depth)
      : base_type{ width, height, depth }
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
      static_assert(Dimension == 2, "MultiArray must be 2D");
      return matrix_view_type {
        reinterpret_cast<
        typename ElementTraits<pixel_type>::pointer>(base_type::data()),
        height(), width()
      };
    }

    inline const_matrix_view_type matrix() const
    {
      static_assert(Dimension == 2, "MultiArray must be 2D");
      return const_matrix_view_type {
        reinterpret_cast<
        typename ElementTraits<pixel_type>::const_pointer>(base_type::data()),
        height(), width()
      };
    }
    //! @}

    template <typename Op>
    inline ImageBase& pixelwise_transform_inplace(Op op)
    {
      for (auto pixel = base_type::begin(); pixel != base_type::end(); ++pixel)
        op(*pixel);
      return *this;
    }

    template <typename Op>
    inline auto pixelwise_transform(Op op) const
      -> Image<decltype(op(std::declval<pixel_type>())), Dimension>
    {
      using PixelType = decltype(op(std::declval<pixel_type>()));
      Image<PixelType, Dimension> dst{ this->sizes() };
      auto src_pixel = this->begin();
      auto dst_pixel = dst.begin();
      for ( ; src_pixel != this->end(); ++src_pixel, ++dst_pixel)
        *dst_pixel = op(*src_pixel);
      return dst;
    }
  };


  //! @brief The image view class.
  template <typename T, int N>
  class ImageView : public ImageBase<MultiArrayView<T, N, ColMajor>>
  {
    using base_type = ImageBase<MultiArrayView<T, N, ColMajor>>;

  public: /* interface */
    using vector_type = typename base_type::vector_type;

    inline ImageView(T *data, const vector_type& sizes)
      : base_type{ data, sizes }
    {
    }
  };


  //! @brief The image class.
  template <typename T, int N>
  class Image : public ImageBase<MultiArray<T, N, ColMajor>>
  {
    using base_type = ImageBase<MultiArray<T, N, ColMajor>>;

  public: /* interface */
    using vector_type = typename base_type::vector_type;

    //! @brief Default constructor.
    Image() = default;

    //! @brief Constructor that takes ownership of data.
    inline explicit Image(T *data, const vector_type& sizes)
      : base_type{ data, sizes }
    {
    }

    //! @{
    //! @brief Constructors with specified sizes.
    inline explicit Image(const vector_type& sizes)
      : base_type{ sizes }
    {
    }

    inline Image(int width, int height)
      : base_type{ width, height }
    {
    }

    inline Image(int width, int height, int depth)
      : base_type{ width, height, depth }
    {
    }
    //! @}

    //! @brief Converts image to the specified pixel format.
    template <typename U>
    inline Image<U, N> convert() const
    {
      auto dst = Image<U, N>{ base_type::sizes() };
      DO::Sara::convert(*this, dst);
      return dst;
    }

    //! @brief Perform custom filtering on the image.
    template <typename Filter, typename... Params>
    inline typename Filter::template ReturnType<Image<T, N>>
    compute(const Params&... params) const
    {
      return Filter{}(*this, params...);
    }
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_CORE_IMAGE_IMAGE_HPP */
