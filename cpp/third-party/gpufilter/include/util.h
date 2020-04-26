/**
 *  @file util.h
 *  @brief Matrix and vector utility classes to facilitate head/tail operations
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 */

#ifndef UTIL_H
#define UTIL_H

//== INCLUDES =================================================================

#include <cassert>
#include <iostream>

#ifdef  __CUDA_ARCH__
#   ifdef assert
#       undef assert
#   endif
#   define assert (void)
#endif

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @class Vector util.h
 *  @ingroup utils
 *  @brief Vector class
 *
 *  Vector class to represent special small vectors, such as the
 *  vector of filter weights \f$a_k\f$ used in forward and reverse
 *  filter computation (see equations 1 and 3 in [Nehab:2011] cited in
 *  alg5()).
 *
 *  @tparam T Vector value type
 *  @tparam N Number of elements
 */
template <class T, int N>
class Vector {
    
public:

    /**
     *  @brief Convert a vector of this class to an stl vector
     *  @return STL Vector
     */
    std::vector<T> to_vector() const {
        return std::vector<T>(&m_data[0], &m_data[0]+size());
    }

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Position to access
     *  @return Value (constant reference) at given position
     */
    __host__ __device__ const T& operator [] ( int i ) const {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     *  @brief Access operator
     *  @param[in] i Position to access
     *  @return Value at given position
     */
    __host__ __device__ T& operator [] ( int i ) {
        assert(i >= 0 && i < size());
        return m_data[i];
    }

    /**
     *  @brief Get size (number of elements) of this vector
     *  @return Vector size
     */
    __host__ __device__ int size() const { return N; }

    /**
     *  @brief Output stream operator
     *  @param[in,out] out Output stream
     *  @param[in] v Vector to output values from
     *  @return Output stream
     */
    friend std::ostream& operator << ( std::ostream& out,
                                       const Vector& v ) {
        out << '[';
        for (int i=0; i<v.size(); ++i) {
            out << v[i];
            if (i < v.size()-1) out << ',';
        }
        return out << ']';
    }

    /**
     *  @brief Add-then-assign operator
     *  @param[in] v Vector to add values from
     *  @return This vector added with the input vector
     */
    __host__ __device__ Vector& operator += ( const Vector& v ) {
        assert(size() == v.size());
#pragma unroll
        for (int j=0; j<size(); ++j)
            m_data[j] += v[j];
        return *this;
    }

    /**
     *  @brief Add operator
     *  @param[in] v Vector to add values from
     *  @return Vector resulting from the addition with the input vector
     */
    __host__ __device__ Vector operator + ( const Vector& v ) const {
        return Vector(*this) += v;
    }

    /**
     *  @brief Multiply-then-assign operator
     *  @param[in] v Vector to multiply values from
     *  @return This vector multiplied with the input vector
     */
    __host__ __device__ Vector &operator *= ( const T& v ) {
#pragma unroll
        for (int j=0; j<size(); ++j)
            m_data[j] *= v;
        return *this;
    }

    /**
     *  @brief Multiply operator
     *  @param[in] v Vector to multiply values from
     *  @return Vector resulting from the multiplication with the input vector
     */
    __host__ __device__ Vector operator * ( const T& v ) const {
        return Vector(*this) *= v;
    }

    /**
     *  @brief Divide-then-assign operator
     *  @param[in] v Vector to divide values from
     *  @return This vector divided with the input vector
     */
    __host__ __device__ Vector& operator /= ( const T& v ) {
#pragma unroll
        for (int j=0; j<size(); ++j)
            m_data[j] /= v;
        return *this;
    }

    /**
     *  @brief Divide operator
     *  @param[in] v Vector to divide values from
     *  @return Vector resulting from the division with the input vector
     */
    __host__ __device__ Vector operator / ( const T& v ) const {
        return Vector(*this) /= v;
    }

    /**
     *  @brief Pointer-access operator (constant)
     *  @return Pointer to the first element of this vector
     */
    __host__ __device__ operator const T* () const { return &m_data[0]; }

    /**
     *  @brief Pointer-access operator
     *  @return Pointer to the first element of this vector
     */
    __host__ __device__ operator T* () { return &m_data[0]; }

private:

    T m_data[N]; ///< Vector values

};

//== CLASS DEFINITION =========================================================

/**
 *  @class Matrix util.h
 *  @ingroup utils
 *  @brief Matrix class
 *
 *  Matrix class to represent special small matrices, such as
 *  \f$A_{FB}\f$ and \f$A_{RB}\f$ described in the paper ([Nehab:2011]
 *  cited in alg5()).
 *
 *  @tparam T Matrix value type
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 */
template <class T, int M, int N=M>
class Matrix {

public:

    /**
     *  @brief Get number of rows of this matrix
     *  @return Number of rows
     */
    __host__ __device__ int rows() const { return M; }

    /**
     *  @brief Get number of columns of this matrix
     *  @return Number of columns
     */
    __host__ __device__ int cols() const { return N; }

    /**
     *  @brief Access (constant) operator
     *  @param[in] i Row of the matrix to access
     *  @return Vector (constant) of the corresponding row
     */
    __host__ __device__ const Vector<T,N>& operator [] ( int i ) const {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     *  @brief Access operator
     *  @param[in] i Row of the matrix to access
     *  @return Vector of the corresponding row
     */
    __host__ __device__ Vector<T,N>& operator [] ( int i ) {
        assert(i >= 0 && i < rows());
        return m_data[i];
    }

    /**
     *  @brief Output stream operator
     *  @param[in,out] out Output stream
     *  @param[in] m Matrix to output values from
     *  @return Output stream
     */
    friend std::ostream& operator << ( std::ostream& out,
                                       const Matrix& m ) {
        out << '[';
        for (int i=0; i<m.rows(); ++i) {
            for (int j=0; j<m.cols(); ++j) {
                out << m[i][j];
                if (j < m.cols()-1) out << ',';
            }
            if (i < m.rows()-1) out << ";\n";
        }
        return out << ']';
    }

    /**
     *  @brief Multiply operator
     *  @param[in] rhs Right-hand-side matrix to multiply
     *  @return Resulting matrix from multiplication
     *  @tparam P Number of rows of other matrix
     *  @tparam Q Number of columns of other matrix
     */
    template <int P, int Q>
    __host__ __device__ Matrix<T,M,Q> operator * ( const Matrix<T,P,Q>& rhs ) const {
        assert(cols()==rhs.rows());
        Matrix<T,M,Q> r;
        for (int i=0; i<r.rows(); ++i) {
            for (int j=0; j<r.cols(); ++j) {
                r[i][j] = m_data[i][0]*rhs[0][j];
                for (int k=1; k<cols(); ++k)
                    r[i][j] += m_data[i][k]*rhs[k][j];
            }
        }
        return r;
    }

    /**
     *  @brief Multiply-then-assign operator
     *  @param[in] val Scalar value to multilpy matrix to
     *  @return Resulting matrix from multiplication
     */
    __host__ __device__ Matrix& operator *= ( T val ) {
#pragma unroll
        for (int i=0; i<rows(); ++i)
#pragma unroll
            for (int j=0; j<cols(); ++j)
                m_data[i][j] *= val;
        return *this;
    }

    /**
     *  @brief Get column j of this matrix
     *  @param[in] j Index of column to get
     *  @return Column j as a vector
     */
    __host__ __device__ Vector<T,M> col( int j ) const {
        Vector<T,M> c;
#pragma unroll
        for (int i=0; i<rows(); ++i)
            c[i] = m_data[i][j];
        return c;
    }

    /**
     *  @brief Set column j of this matrix
     *  @param[in] j Index of column to set
     *  @param[in] c Vector to place in matrix column
     */
    __host__ __device__ void set_col( int j,
                                      const Vector<T,M>& c ) {
#pragma unroll
        for (int i=0; i<rows(); ++i)
            m_data[i][j] = c[i];
    }

    /**
     *  @brief Multiply operator
     *  @param[in] m Matrix to multiply
     *  @param[in] val Scalar value to multiply matrix to
     *  @return Resulting matrix from the multiplication
     */
    __host__ __device__ friend Matrix operator * ( const Matrix& m,
                                                   T val ) {
        return Matrix(m) *= val;
    }

    /**
     *  @brief Multiply operator
     *  @param[in] val Scalar value to multiply matrix to
     *  @param[in] m Matrix to multiply
     *  @return Resulting matrix from the multiplication
     */
    __host__ __device__ friend Matrix operator * ( T val,
                                                   const Matrix& m ) {
        return operator*(m,val);
    }

    /**
     *  @brief Add-then-assign operator
     *  @param[in] rhs Right-hand-side matrix in addition
     *  @return Resulting matrix from the addition
     */
    __host__ __device__ Matrix& operator += ( const Matrix& rhs ) {
#pragma unroll
        for (int i=0; i<rows(); ++i)
#pragma unroll
            for (int j=0; j<cols(); ++j)
                m_data[i][j] += rhs[i][j];
        return *this;
    }

    /**
     *  @brief Add operator
     *  @param[in] lhs Left-hand-side matrix in addition
     *  @param[in] rhs Right-hand-side matrix in addition
     *  @return Resulting matrix from the addition
     */
    __host__ __device__ friend Matrix operator + ( const Matrix& lhs,
                                                   const Matrix& rhs ) {
        return Matrix(lhs) += rhs;
    }

    /**
     *  @brief Subtract-then-assign operator
     *  @param[in] rhs Right-hand-side matrix in subtraction
     *  @return Resulting matrix from the subtraction
     */
    __host__ __device__ Matrix& operator -= ( const Matrix& rhs ) {
#pragma unroll
        for (int i=0; i<rows(); ++i)
#pragma unroll
            for (int j=0; j<cols(); ++j)
                m_data[i][j] -= rhs[i][j];
        return *this;
    }

    /**
     *  @brief Subtract operator
     *  @param[in] lhs Left-hand-side matrix in subtraction
     *  @param[in] rhs Right-hand-side matrix in subtraction
     *  @return Resulting matrix from the subtraction
     */
    __host__ __device__ friend Matrix operator - ( const Matrix& lhs,
                                                   const Matrix& rhs ) {
        return Matrix(lhs) -= rhs;
    }

private:

    Vector<T,N> m_data[M]; ///< Matrix values

};

//== IMPLEMENTATION ===========================================================

//-- Multiply -----------------------------------------------------------------

/**
 *  @relates Matrix
 *  @brief Multiply vector-to-matrix operator
 *  @param[in] v Vector to multiply the matrix
 *  @param[in] m Matrix to be multiplied by vector
 *  @return Resulting vector from the multiplication
 */
template <class T, int M, int N>
__host__ __device__ Vector<T,N> operator * ( const Vector<T,M>& v,
                                             const Matrix<T,M,N>& m ) {
    assert(v.size() == m.rows());
    Vector<T,N> r;
#pragma unroll
    for (int j=0; j<m.cols(); ++j) {
        r[j] = v[0]*m[0][j];
#pragma unroll
        for (int i=1; i<m.rows(); ++i)
            r[j] += v[i]*m[i][j];
    }
    return r;
}

//-- Basics -------------------------------------------------------------------

/**
 *  @relates Vector
 *  @brief Instantiate a vector with zeros
 *  @tparam T Vector value type
 *  @tparam N Number of elements
 */
template <class T, int N>
__host__ __device__ Vector<T,N> zeros() {
    Vector<T,N> v;
    if (N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for (int j=0; j<v.size(); ++j)
            v[j] = T();
#else
        std::fill(&v[0], &v[N-1]+1, T());
#endif
    }
    return v; // I'm hoping that RVO will kick in
}

/**
 *  @relates Matrix
 *  @brief Instantiate a matrix with zeros
 *  @tparam T Matrix value type
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 */
template <class T, int M, int N>
__host__ __device__ Matrix<T,M,N> zeros() {
    Matrix<T,M,N> mat;
    if (M>0 && N>0)
    {
#if __CUDA_ARCH__
#pragma unroll
        for (int i=0; i<mat.rows(); ++i)
#pragma unroll
            for (int j=0; j<mat.cols(); ++j)
                mat[i][j] = T();
#else
        std::fill(&mat[0][0], &mat[M-1][N-1]+1, T());
#endif
    }
    return mat; // I'm hoping that RVO will kick in
}

/**
 *  @relates Matrix
 *  @brief Instantiate an identity matrix
 *  @tparam T Matrix value type
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 */
template <class T, int M, int N>
Matrix<T,M,N> identity() {
    Matrix<T,M,N> mat;
    for (int i=0; i<M; ++i)
        for (int j=0; j<N; ++j)
            mat[i][j] = i==j ? 1 : 0;
    return mat;
}

/**
 *  @relates Matrix
 *  @brief Instantiate the transposed version of a given matrix
 *  @param[in] m Given matrix
 *  @tparam T Matrix value type
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 */
template <class T, int M, int N>
__host__ __device__ Matrix<T,N,M> transp( const Matrix<T,M,N>& m ) {
    Matrix<T,N,M> tm;
#pragma unroll
    for (int i=0; i<m.rows(); ++i)
#pragma unroll
        for (int j=0; j<m.cols(); ++j)
            tm[j][i] = m[i][j];
    return tm;
}

//-- Forward ------------------------------------------------------------------

/**
 *  @relates Vector
 *  @brief Computes the \a forward operator on vectors (in-place)
 *
 *  For more information see fwd().
 *
 *  @param[in] p Prologue \f$R\f$ vector (\f$R\f$ is the filter order)
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam N Number of elements
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int N, int R>
void fwd_inplace( const Vector<T,R>& p,
                  Vector<T,N>& b,
                  const Vector<T,R+1>& w ) {
    for (int j=0; j<b.size(); ++j) {
        b[j] *= w[0];
        for (int k=1; k<w.size(); ++k) {
            if (j-k < 0)
                b[j] -= p[p.size()+j-k]*w[k]; // use data from prologue
            else
                b[j] -= b[j-k]*w[k];
        }
    }
}

/**
 *  @relates Matrix
 *  @brief Computes the \a forward operator on matrices (in-place)
 *
 *  For more information see rev().
 *
 *  @param[in] p Prologue \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
void fwd_inplace( const Matrix<T,M,R>& p,
                  Matrix<T,M,N>& b,
                  const Vector<T,R+1>& w ) {
    for (int i=0; i<b.rows(); ++i)
        fwd_inplace(p[i], b[i], w);
}

/**
 *  @relates Matrix
 *  @brief Computes the \a forward operator on matrices
 *
 *  Computes the matrix resulting from applying the \a forward
 *  operator \f$F\f$ (causal filter) given a prologue \f$R \times N\f$
 *  matrix \f$p\f$ (i.e. initial conditions) and an input \f$M \times
 *  N\f$ matrix \f$b\f$ (where M is the number of rows \f$h\f$ and N
 *  is the number of columns \f$w\f$ as described in section 2 of
 *  [Nehab:2011] cited in alg5()).  The resulting matrix is \f$M
 *  \times N\f$ and it has the same size as the input \f$b\f$.
 *
 *  @param[in] p Prologue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwd( const Matrix<T,M,R>& p,
                   const Matrix<T,M,N>& b, 
                   const Vector<T,R+1>& w) {
    Matrix<T,M,N> fb = b;
    fwd_inplace(p, fb, w);
    return fb;
}

/**
 *  @relates Matrix
 *  @overload
 *  @brief Computes the \a forward operator on matrices (in-place)
 *
 *  For more information see fwd().
 *
 *  @param[in] p Prologue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
void fwd_inplace( const Matrix<T,R,N>& p,
                  Matrix<T,M,N>& b,
                  const Vector<T,R+1>& w ) {
    b = fwd(p, b, w);
}

/**
 *  @relates Matrix
 *  @brief Computes the \a forward-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  forward-transposed operator \f$F^{T}\f$ (causal filter on rows)
 *  given a prologue \f$M \times R\f$ matrix \f$p^{T}\f$ (i.e. initial
 *  conditions) and an input \f$M \times N\f$ matrix \f$b\f$ (where M
 *  is the number of rows \f$h\f$ and N is the number of columns
 *  \f$w\f$ as described in section 2 of [Nehab:2011] cited in
 *  alg5()).  The resulting matrix is \f$M \times N\f$ and it has the
 *  same size as the input \f$b\f$.
 *
 *  @param[in] pT Prologue transposed \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> fwdT( const Matrix<T,R,N>& pT,
                    const Matrix<T,M,N>& b, 
                    const Vector<T,R+1>& w ) {
    return transp(fwd(transp(pT), transp(b), w));
}

//-- Reverse ------------------------------------------------------------------

/**
 *  @relates Vector
 *  @brief Computes the \a reverse operator on vectors (in-place)
 *
 *  For more information see rev().
 *
 *  @param[in,out] b In(out)put \f$N\f$ vector
 *  @param[in] e Epilogue \f$R\f$ vector (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam N Number of elements
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int N, int R>
void rev_inplace( Vector<T,N>& b,
                  const Vector<T,R>& e,
                  const Vector<T,R+1>& w ) {
    for (int j=b.size()-1; j>=0; --j) {
        b[j] *= w[0];
        for (int k=1; k<w.size(); ++k) {
            if (j+k >= b.size())
                b[j] -= e[j+k-b.size()]*w[k]; // use data from epilogue
            else
                b[j] -= b[j+k]*w[k];
        }
    }
}

/**
 *  @relates Matrix
 *  @brief Computes the \a reverse operator on matrices (in-place)
 *
 *  For more information see rev().
 *
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
void rev_inplace( Matrix<T,M,N>& b,
                  const Matrix<T,M,R>& e,
                  const Vector<T,R+1>& w ) {
    for (int i=0; i<b.rows(); ++i)
        rev_inplace(b[i], e[i], w);
}

/**
 *  @relates Matrix
 *  @brief Computes the \a reverse operator on matrices
 *
 *  Computes the matrix resulting from applying the \a reverse
 *  operator \f$R\f$ (anticausal filter) given an epilogue \f$R \times
 *  N\f$ matrix \f$e\f$ (i.e. initial conditions) and an input \f$M
 *  \times N\f$ matrix \f$b\f$ (where M is the number of rows \f$h\f$
 *  and N is the number of columns \f$w\f$ as described in section 2
 *  of [Nehab:2011] cited in alg5()).  The resulting matrix is \f$M
 *  \times N\f$ and it has the same size as the input \f$b\f$.
 *
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$M \times R\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @return Matrix resulting from applying the \a reverse operator
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> rev( const Matrix<T,M,N>& b,
                   const Matrix<T,M,R>& e, 
                   const Vector<T,R+1>& w ) {
    Matrix<T,M,N> rb = b;
    rev_inplace(rb, e, w);
    return rb;
}

/**
 *  @relates Matrix
 *  @overload
 *  @brief Computes the \a reverse operator on matrices (in-place)
 *
 *  For more information see rev().
 *
 *  @param[in,out] b In(out)put \f$M \times N\f$ matrix
 *  @param[in] e Epilogue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
void rev_inplace( Matrix<T,M,N>& b,
                  const Matrix<T,R,N>& e,
                  const Vector<T,R+1>& w ) {
    b = rev(b, e, w);
}

/**
 *  @relates Matrix
 *  @brief Computes the \a reverse-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a
 *  reverse-transposed operator \f$R^{T}\f$ (anticausal filter on
 *  rows) given an epilogue \f$M \times R\f$ matrix \f$e^{T}\f$
 *  (i.e. initial conditions) and an input \f$M \times N\f$ matrix
 *  \f$b\f$ (where M is the number of rows \f$h\f$ and N is the number
 *  of columns \f$w\f$ as described in section 2 of [Nehab:2011] cited
 *  in alg5()).  The resulting matrix is \f$M \times N\f$ and it has
 *  the same size as the input \f$b\f$.
 *
 *  @param[in] b Input \f$M \times N\f$ matrix
 *  @param[in] eT Epilogue \f$R \times N\f$ matrix (\f$R\f$ is the filter order)
 *  @param[in] w Filter weights with \f$R+1\f$ size (\f$R\f$ feedback coefficients)
 *  @tparam M Number of rows
 *  @tparam N Number of columns
 *  @tparam R Number of feedback coefficients
 *  @tparam T Matrix value type
 */
template <class T, int M, int N, int R>
Matrix<T,M,N> revT( const Matrix<T,M,N>& b,
                    const Matrix<T,R,N>& eT,
                    const Vector<T,R+1>& w ) {
    return transp(rev(transp(b), transp(eT), w));
}

//-- Head ---------------------------------------------------------------------

/**
 *  @relates Matrix
 *  @brief Computes the \a head operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head operator
 *  \f$H\f$ given an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$R \times N\f$ submatrix in the same shape
 *  and position as the column-epilogue of the input matrix,
 *  considering filter order \f$R\f$ (see [Nehab:2011] cited in alg5()
 *  function).  The following image illustrates the concept:
 *
 *  @image html blocks-2d.png "2D Block Notation"
 *  @image latex blocks-2d.eps "2D Block Notation" width=\textwidth
 *
 *
 *  2D block notation showing a block and its boundary data from
 *  adjacent blocks.  Note that the column-epilogue \f$E_{m+1,n}\f$
 *  has to be extracted from the next block \f$B_{m+1,n}\f$ in column
 *  \f$n\f$ using the head operator on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<T,M,R> head( const Matrix<T,M,N>& mat ) {
    assert(mat.cols() >= R);
    Matrix<T,M,R> h;
    for (int j=0; j<R; ++j)
        for (int i=0; i<mat.rows(); ++i)
            h[i][j] = mat[i][j];
    return h;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a head-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a head-transposed
 *  operator \f$H^{T}\f$ given an input \f$M \times N\f$ matrix
 *  \f$mat\f$.  The operator extracts the \f$M \times R\f$ submatrix
 *  in the same shape and position as the row-epilogue of the input
 *  matrix, considering filter order \f$R\f$ (see [Nehab:2011] cited
 *  in alg5() function).
 *
 *  Note that, as shown in figure in head() function, the row-epilogue
 *  \f$E^{T}_{m,n+1}\f$ has to be extracted from the next block
 *  \f$B_{m,n+1}\f$ in row \f$m\f$ using the head-transposed operator
 *  on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of columns to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<T,R,N> headT( const Matrix<T,M,N>& mat ) {
    return transp(head<R>(transp(mat)));
}

//-- Tail ---------------------------------------------------------------------

/**
 *  @relates Matrix
 *  @brief Computes the \a tail operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail operator
 *  \f$T\f$ given an input \f$M \times N\f$ matrix \f$mat\f$.  The
 *  operator extracts the \f$R \times N\f$ submatrix in the same shape
 *  and position as the column-prologue of the input matrix,
 *  considering filter order \f$R\f$ (see [Nehab:2011] cited in alg5()
 *  function).
 *
 *  Note that, as shown in figure in head() function, the
 *  column-prologue \f$P_{m-1,n}\f$ has to be extracted from the
 *  previous block \f$B_{m-1,n}\f$ in column \f$n\f$ using the tail
 *  operator on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<T,M,R> tail( const Matrix<T,M,N>& mat ) {
    assert(mat.cols() >= R);
    Matrix<T,M,R> t;
    for (int j=0; j<R; ++j)
        for (int i=0; i<mat.rows(); ++i)
            t[i][j] = mat[i][mat.cols()-R+j];
    return t;
}

/**
 *  @relates Matrix
 *  @brief Computes the \a tail-transposed operator on matrices
 *
 *  Computes the matrix resulting from applying the \a tail-transposed
 *  operator \f$T^{T}\f$ given an input \f$M \times N\f$ matrix
 *  \f$mat\f$.  The operator extracts the \f$M \times R\f$ submatrix
 *  in the same shape and position as the row-prologue of the input
 *  matrix, considering filter order \f$R\f$ (see [Nehab:2011] cited
 *  in alg5() function).
 *
 *  Note that, as shown in figure in head() function, the row-prologue
 *  \f$P^{T}_{m,n-1}\f$ has to be extracted from the previous block
 *  \f$B_{m,n-1}\f$ in row \f$m\f$ using the tail-transposed operator
 *  on that block.
 *
 *  @param[in] mat Input \f$M \times N\f$ matrix
 *  @tparam R Number of rows to extract (feedback coefficients)
 *  @tparam M Number of rows in input matrix
 *  @tparam N Number of columns in input matrix
 *  @tparam T Matrix value type
 */
template <int R, int M, int N, class T>
Matrix<T,R,N> tailT( const Matrix<T,M,N>& mat ) {
    return transp(tail<R>(transp(mat)));
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // UTIL_H
//=============================================================================
