#pragma once

#include <DO/Sara/Core/Expression/Expression.hpp>

#include <string>


namespace DO::Sara {

  template <typename E>
  using Expression = sara::expression::Expression<E>;


  struct Symbol : public Expression<Symbol>
  {
    const std::string name;

    Symbol(const std::string& name)
      : name{name}
    {
    }

    auto operator<(const Symbol& other) const -> bool
    {
      return name < other.name;
    }

    auto operator==(const Symbol& other) const -> bool
    {
      return name == other.name;
    }
  };

  template <typename T>
  struct Placeholder : Symbol
  {
    T value;
  };

  template <typename T>
  struct Variable : Placeholder<T>
  {
  };

  template <typename T>
  struct Constant : Placeholder<T>
  {
    const T value;
  };

} /* namespace DO::Sara */
