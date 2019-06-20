#pragma once

#include <DO/Sara/Core/Expression/ForwardDeclarations.hpp>
#include <DO/Sara/Core/Expression/TypeCalculation.hpp>
#include <DO/Sara/Core/Expression/Arithmetics.hpp>


namespace sara::expression {

  template <typename Derived>
  struct Expression
  {
    using derived_type = Derived;

    //! @{
    //! @brief derived object
    inline auto& derived() & noexcept
    {
      return static_cast<derived_type&>(*this);
    }

    inline const auto& derived() const& noexcept
    {
      return static_cast<const derived_type&>(*this);
    }

    inline auto&& derived() && noexcept
    {
      return static_cast<derived_type&&>(*this);
    }
    //! @}

    template <typename X>
    inline auto eval() const&
    {
      return derived().eval();
    }

    //! @{
    //! @brief Addition operation.
    template <typename R>
    inline auto operator+(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return AddXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator+(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return AddXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator+(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return AddXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}

    //! @{
    //! @brief Subtraction operation.
    template <typename R>
    inline auto operator-(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator-(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator-(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}

    //! @{
    //! @brief Multiplication operation.
    template <typename R>
    inline auto operator*(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return MulXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator*(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return MulXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator*(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return MulXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}


    //! @{
    //! @brief Division operation.
    template <typename R>
    inline auto operator/(R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return DivXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator/(R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return MulXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator/(R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return DivXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}


    //! @{
    //! @brief Subscript operation.
    template <typename R>
    inline auto operator[](R&& rhs) & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubscriptXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator[](R&& rhs) const & noexcept
    {
      using lhs_type = decltype(derived());
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubscriptXpr<lhs_type, rhs_type>{
          derived(),            //
          std::forward<R>(rhs)  //
      };
    }

    template <typename R>
    inline auto operator[](R&& rhs) && noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<R&&>;

      return SubscriptXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),  //
          std::forward<R>(rhs)                //
      };
    }
    //! @}


    //! Evaluate function at variable x.
    template <typename X>
    inline auto operator()(X&& x) const noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<X&&>;

      return FunXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),
          std::forward<rhs_type>(x),
      };
    }


    //! Function composition.
    template <typename F>
    inline auto circle(F&& f) const noexcept
    {
      using lhs_type = std::remove_reference_t<decltype(derived())>;
      using rhs_type = calculate_expr_type_t<F&&>;

      return FunComposeXpr<lhs_type, rhs_type>{
          std::forward<lhs_type>(derived()),
          std::forward<rhs_type>(f),
      };
    }
  };

} /* namespace sara::expression */
