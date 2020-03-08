/**
 *  @file extension.h
 *  @brief Initial condition definitions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date December, 2011
 */

#ifndef EXTENSION_H
#define EXTENSION_H

//== INCLUDES =================================================================

#include <cmath>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== ENUMERATION ==============================================================

/**
 *  @ingroup utils
 *  @brief Enumerates possible initial conditions for 2D-image access
 */
enum initcond {
    zero, ///< Zero-border (outside image everything is zero)
    clamp, ///< Clamp-to-border (image border continues forever)
    repeat, ///< Repeat (image repeats itself forever)
    mirror ///< Mirror (image repeats itself mirrored forever)
};

//== CLASS DEFINITION =========================================================

/**
 *  @struct _clamp extension.h
 *  @ingroup utils
 *  @brief Access pattern: clamp-to-border
 */
struct _clamp {
    /**
     *  @brief Functor operator
     *  @param t Float value in range [0,1]
     *  @return Correct float parameter with clamp-to-border
     */
    float operator () ( float t ) const {
        if( t < 0.f ) return 0.f;
        else if( t > 1.f ) return 1.f;
        else return t;
    }
    /**
     *  @brief Functor operator
     *  @param i Integer index in range [0,n)
     *  @param n Defines range [0,n)
     *  @return Correct integer index with clamp-to-border
     */
    int operator () ( int i, int n ) const {
        if( i < 0 ) return 0;
        else if( i >= n ) return n-1;
        else return i;
    }
};

//== CLASS DEFINITION =========================================================

/**
 *  @struct _repeat extension.h
 *  @ingroup utils
 *  @brief Access pattern: repeat
 */
struct _repeat {
    /**
     *  @brief Functor operator
     *  @param t Float value in range [0,1]
     *  @return Correct float parameter with repeat
     */
    float operator () ( float t ) const {
        t = fmodf(t, 1.f);
        return t < 0.f ? t + 1.f : t;
    }
    /**
     *  @brief Functor operator
     *  @param i Integer index in range [0,n)
     *  @param n Defines range [0,n)
     *  @return Correct integer index with repeat
     */
    int operator () ( int i, int n ) const {
        if( i >= 0 ) return i % n;
        else return (n-1) - ((-i-1) % n);
    }
};

//== CLASS DEFINITION =========================================================

/**
 *  @struct _mirror extension.h
 *  @ingroup utils
 *  @brief Access pattern: mirror
 */
struct _mirror {
    /**
     *  @brief Functor operator
     *  @param t Float value in range [0,1]
     *  @return Correct float parameter with mirror
     */
    float operator () ( float t ) const {
        t = fabs(fmodf(t, 2.f));
        return t > 1.f ? 2.f - t : t;
    }
    /**
     *  @brief Functor operator
     *  @param i Integer index in range [0,n)
     *  @param n Defines range [0,n)
     *  @return Correct integer index with mirror
     */
    int operator () ( int i, int n ) const {
        _repeat r;
        i = r(i, 2*n); 
        if( i >= n ) return i = (2*n)-i-1;
        else return i;
    }
};

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup utils
 *  @brief Look in an input at a given index range
 *
 *  This function looks in an input array at an arbitrary index
 *  (possibly outside the input array) using a given initial condition
 *  and a pitch value for arrays with 2D geometry.
 *
 *  @param[in] in Input array
 *  @param[in] i Index to look at
 *  @param[in] n Input range [0,n)
 *  @param[in] ic Initial condition
 *  @param[in] p Pitch (e.g. image width represented by the input array)
 *  @return Value at given position
 *  @tparam T Input array value type
 */
template< class T >
T lookat( const T *in,
          const int& i,
          const int& n,
          const initcond& ic,
          const int& p = 1 ) {
    if( !in ) return (T)0;
    switch( ic ) {
    case zero:
        if( i < 0 ) return (T)0;
        else if( i >= n ) return (T)0;
        else return in[i*p];
    case clamp:
        _clamp c;
        return in[ c(i, n)*p ];
    case repeat:
        _repeat r;
        return in[ r(i, n)*p ];
    case mirror:
        _mirror m;
        return in[ m(i, n)*p ];
    }
    return (T)0;
}

/**
 *  @ingroup utils
 *  @overload
 *  @brief Look in an image at a given position
 *
 *  This function looks in an input image at an arbitrary position
 *  (possibly outside the image) using a given initial condition and
 *  the 2D image size.
 *
 *  @param[in] img Input image
 *  @param[in] i Row index
 *  @param[in] j Column index
 *  @param[in] h Image height
 *  @param[in] w Image width
 *  @param[in] ic Initial condition
 *  @return Value at given position
 *  @tparam T Input image value type
 */
template< class T >
T lookat( const T *img,
          const int& i,
          const int& j,
          const int& h,
          const int& w,
          const initcond& ic ) {
    if( !img ) return (T)0;
    switch( ic ) {
    case zero:
        if( i < 0 or j < 0 ) return (T)0;
        else if( i >= h or j >= w ) return (T)0;
        else return img[i*w+j];
    case clamp:
        _clamp c;
        return img[ c(i, h)*w + c(j, w) ];
    case repeat:
        _repeat r;
        return img[ r(i, h)*w + r(j, w) ];
    case mirror:
        _mirror m;
        return img[ m(i, h)*w + m(j, w) ];
    }
    return (T)0;
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // EXTENSION_H
//=============================================================================
