#pragma once

#include <drafts/Expression/ForwardDeclarations.hpp>

#include <functional>
#include <string>


namespace sara::expression {

  template <typename T>
  struct Terminal : Expression<Terminal<T>>
  {
    using decayed_type = Terminal<std::decay_t<T>>;

    inline Terminal() = default;

    inline constexpr Terminal(T v)
      : value{v}
    {
    }

    auto eval() const & noexcept -> const T&
    {
      return value;
    }

    auto eval() && -> T
    {
      return value;
    }

    template <typename X>
    auto eval_at(X&& x) const
    {
      return value(x);
    }

    template <typename Index>
    auto at(Index&& i) const&
    {
      return value[i];
    }

    T value;
  };


  template <typename T>
  struct Terminal<T(T)> : Expression<Terminal<T(T)>>
  {
    using decayed_type = Terminal<std::decay_t<T(T)>>;

    inline constexpr Terminal(std::function<T(T)> v)
      : value{v}
    {
    }

    std::function<T(T)> value;
  };

  template <typename T>
  inline constexpr auto make_terminal(T&& t) -> Terminal<T>
  {
    return {std::forward<T>(t)};
  }

} /* namespace sara::expression */
