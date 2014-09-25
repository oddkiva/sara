#ifndef DO_CORE_PIXEL_CHANNELCONVERSION_HPP
#define DO_CORE_PIXEL_CHANNELCONVERSION_HPP


#include <DO/Core/StaticAssert.hpp>
#include <DO/Core/EigenExtension.hpp>


// Channel conversion from a type to another.
namespace DO {

  //! \brief Return maximum value for channel of type 'T'.
  template <typename T>
  inline T channel_min_value()
  {
    using std::numeric_limits;
    return numeric_limits<T>::is_integer ? numeric_limits<T>::min() : T(0);
  }

  //! \brief Return maximum value for channel of type.
  template <typename T>
  inline T channel_max_value()
  {
    using std::numeric_limits;
    return numeric_limits<T>::is_integer ? numeric_limits<T>::max() : T(1);
  }

  //! \brief Convert integral channel value to floating-point value.
  template <typename Int, typename Float>
  inline Float float_normalized_channel(Int src)
  {
    DO_STATIC_ASSERT(
      std::numeric_limits<Int>::is_integer,
      CHANNEL_CONVERSION_MUST_BE_FROM_INTEGER_TYPE_TO_FLOATING_POINT_TYPE);

    using std::numeric_limits;
    const Float float_min = static_cast<Float>(numeric_limits<Int>::min());
    const Float float_max = static_cast<Float>(numeric_limits<Int>::max());
    const Float float_range = float_max - float_min; 
    return (static_cast<Float>(src) - float_min) / float_range;
  }
  
  //! \brief Convert floating-point channel value to integer value.
  template <typename Int, typename Float>
  inline Int int_rescaled_channel(Float src)
  {
    DO_STATIC_ASSERT(
      std::numeric_limits<Int>::is_integer,
      CHANNEL_CONVERSION_MUST_BE_FROM_FLOATING_POINT_TYPE_TO_INTEGER_TYPE);

    using std::numeric_limits;
    const Float float_min = static_cast<Float>(numeric_limits<Int>::min());
    const Float float_max = static_cast<Float>(numeric_limits<Int>::max());
    const Float float_range = float_max - float_min; 
    src = float_min + src * float_range;
    
    const Float delta_max = std::abs(src-float_max)/float_range;
    const Float delta_min = std::abs(src-float_min)/float_range;
    const Float eps = sizeof(Float) == 4 ?
      Float(1e-5) : // i.e., if 'Float' == 'float'.
      Float(1e-9);  // i.e., if 'Float' == 'double'.

    Int dst;
    if (delta_max <= eps)
      dst = std::numeric_limits<Int>::max();
    else if (delta_min <= eps)
      dst = std::numeric_limits<Int>::min();
    else
      dst = static_cast<Int>(floor(src + 0.5));
    return dst;
  }

} /* namespace DO */


// Unified API for channel conversion.
namespace DO {

  //! \brief Convert a double gray value to a float gray value. 
  inline void convert_channel(double src, float& dst)
  {
    dst = static_cast<float>(src);
  }

  //! \brief Convert a float gray value to a double gray value. 
  inline void convert_channel(float src, double& dst)
  {
    dst = static_cast<double>(src);
  }

  //! \brief Convert an integer gray value to a float gray value.
  template <typename Int>
  inline void convert_channel(Int src, float& dst)
  {
    dst = float_normalized_channel<Int, float>(src);
  }

  //! \brief Convert an integer gray value to a double gray value.
  template <typename Int>
  inline void convert_channel(Int src, double& dst)
  {
    dst = float_normalized_channel<Int, double>(src);
  }

  //! \brief Convert a float gray value to an integer gray value.
  template <typename Int>
  inline void convert_channel(float src, Int& dst)
  {
    dst = int_rescaled_channel<Int, float>(src);
  }

  //! \brief Convert a double gray value to a integer gray value.
  template <typename Int>
  inline void convert_channel(double src, Int& dst)
  {
    dst = int_rescaled_channel<Int, double>(src);
  }

  //! \brief Convert an integer gray value to another one.
  template <typename SrcInt, typename DstInt>
  inline void convert_channel(SrcInt src, DstInt& dst)
  {
    dst =
      int_rescaled_channel<DstInt, double> (
      float_normalized_channel<SrcInt, double>(src) );
  }

  //! \brief Convert channels from a pixel vector to another pixel vector.
  template <typename T, typename U, int N>
  inline void convert_channel(const Matrix<T, N, 1>& src, Matrix<U, N, 1>& dst)
  {
    for (int i = 0; i < N; ++i)
      convert_channel(src[i], dst[i]);
  }

}


#endif /* DO_CORE_PIXEL_COLORSPACE_HPP */