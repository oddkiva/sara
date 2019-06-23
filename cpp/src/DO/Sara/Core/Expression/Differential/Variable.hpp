#pragma once

#include <type_traits>


namespace sara::expression {

  template <typename T, int ID>
  struct Variable
  {
    static constexpr auto name = ID;

    inline Variable() = default;

    inline constexpr Variable(T v)
      : value{v}
    {
    }

    inline operator T&() & noexcept
    {
      return value;
    }

    inline operator const T&() const& noexcept
    {
      return value;
    }

    inline operator T() && noexcept
    {
      return value;
    }

    T value;
  };


  namespace abc {

    template <typename T>
    using x = Variable<T, 'x'>;

    template <typename T>
    using y = Variable<T, 'y'>;

    template <typename T>
    using z = Variable<T, 'z'>;

    template <typename T>
    using X = Variable<T, 'X'>;

    template <typename T>
    using Y = Variable<T, 'Y'>;

    template <typename T>
    using Z = Variable<T, 'Z'>;

  }  // namespace abc

} /* sara::expression */
