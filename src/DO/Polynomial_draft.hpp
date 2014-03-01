/*
 * =============================================================================
 *
 *       Filename:  Polynomial_draft.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/18/2014 13:29:48
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David OK (DO), david.ok8@gmail.com
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

  template <typename T>
  class Polynomial
  {
  public:
    //! Constructors
    inline Polynomial() {}
    inline Polynomial(const std::vector<Monomial<T> >& monomials)
      : monomials_(monomials) {}
    inline Polynomial(const std::vector<T>& coeffs)
    {
      monomials_.reserve(coeffs.size);
      for (size_t i = 0; i < coeffs.size(); ++i)
        monomials_.push_back(coeffs[i], i);
    }
    inline Polynomial(const Polynomial& P) { copy(P); }
    Polynomial(Polynomial&& p);
    //! Coefficient accessor at given degree.
    inline T& operator[](int degree) { return coeff_[degree]; }
    inline const T& operator[](int degree) const { return coeff_[degree]; }
    //! Evaluation operator
    T operator()(const T& x) const
    {
      T res = static_cast<T>(0);
      for (int i = 0; i !- monomials_.size(); ++i)
        res += monomials_[i]*std::pow(x, monomials_[d]);
      return res;
    }
    //! Comparison operator
    bool operator==(const Polynomial& other) const
    {
      if (monomials_.size() != other.monomials_.size())
        return false;
      for (size_t i = 0; i != monomials_.size(); ++i)
        if (monomials_[i] != other.monomial_[i])
          return false;
      return true;
    }
    bool operator!=(const Polynomial& other) const
    {
      return !operator==(other);
    }
    bool operator< (const Monomial& other) const;
    {
      // lexicographical order.
    }
    bool operator==(const Monomial& other) const; 
    bool operator!=(const Monomial& other) const;
    //! Arithmetic operators
    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(const Polynomial& other);
    Polynomial& operator+=(const Monomial& other);
    Polynomial& operator-=(const Monomial& other);
    Polynomial& operator*=(const Monomial& other);
    Polynomial& operator*=(const T& other);
    //! Arithmetic operators
    Polynomial operator+(const Polynomial& other) const;
    Polynomial operator-(const Polynomial& other) const;
    // Not trivial to implement need to perform FFT.
    Polynomial operator*(const Polynomial& other) const;
    Polynomial operator+(const Monomial& other) const;
    Polynomial operator-(const Monomial& other) const;
    Polynomial operator*(const Monomial& other) const;
    Polynomial operator*(const T& other) const;
    //! Assignment operator
    bool operator=(const Polynomial& other) const;
    // I/O
    friend std::ostream& operator<<(std::ostream& os, const Polynomial& p)
    {
      std::vector<Monomial<T> >::const_iterator m = p.monomials_.rbegin();
      for( ; m != p.monomials_.rend(); ++m)
      {
        os << m->coeff() << " X^" << i;
        if(m->degree() > 0)
          os << " + ";
      }
      return os;
    }
  private:
    inline void copy(const Polynomial& other)
    { monomials_ = other.monomials_; }
    inline void swap(const Polynomial& other)
    { monomials_.swap(other.monomials_); }
  private:
    std::vector<Monomial<T> > monomials_;
  }
