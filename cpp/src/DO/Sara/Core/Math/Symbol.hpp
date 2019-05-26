#pragma once

#include <string>


namespace DO { namespace Sara {

  class Expression
  {
  public:
    Expression() = default;
  };


  struct Symbol : public Expression
  {
    const std::string name;
    const bool is_variable;

    Symbol(const std::string& name, bool is_variable)
      : name{name}
      , is_variable{is_variable}
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

  auto variable(const std::string& name) -> Symbol
  {
    return {name, true};
  }

  auto one() -> Symbol
  {
    return {"1", false};
  }

  auto zero() -> Symbol
  {
    return {"0", false};
  }

} /* namespace Sara */
} /* namespace DO */


namespace DO { namespace Sara { namespace v2 {

  template <typename E>
  class Expression
  {
  public:
    Expression() = default;
  };


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


} /* namespace v2 */
} /* namespace Sara */
} /* namespace DO */
