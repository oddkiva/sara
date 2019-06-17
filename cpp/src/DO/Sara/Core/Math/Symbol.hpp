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

  inline auto variable(const std::string& name) -> Symbol
  {
    return {name, true};
  }

  inline auto one() -> Symbol
  {
    return {"1", false};
  }

  inline auto zero() -> Symbol
  {
    return {"0", false};
  }

} /* namespace Sara */
} /* namespace DO */
