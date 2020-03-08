/**
 *  @file cpuground.h
 *  @brief CPU Groundtruth Recursive Filtering functions
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date October, 2010
 */

#ifndef CPUFILTER_H
#define CPUFILTER_H

//== INCLUDES =================================================================

#include <cmath>

#include <defs.h>
#include <extension.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Image --------------------------------------------------------------------

/**
 *  @ingroup api_cpu
 *  @brief Extend an image to consider initial condition outside
 *
 *  Given an input 2D image extend it including a given initial
 *  condition for outside access.
 *
 *  @param[out] ext_img The extended 2D image (to be allocated)
 *  @param[out] ext_w Width of the extended image
 *  @param[out] ext_h Height of the extended image
 *  @param[in] img The 2D image to extend
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] extb Extension (in blocks) to consider outside image
 *  @param[in] ic Initial condition (for outside access)
 *  @tparam T Image value type
 */
template< class T >
void extend_image( T *& ext_img,
                   int& ext_w,
                   int& ext_h,
                   const T *img,
                   const int& w,
                   const int& h,
                   const int& extb,
                   const initcond& ic ) {
    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, w, h, extb );
    ext_w = w+bleft+bright;
    ext_h = h+btop+bbottom;
    ext_img = new float[ ext_w * ext_h ];
    for (int i = -btop; i < h+bbottom; ++i) {
        for (int j = -bleft; j < w+bright; ++j) {
            ext_img[(i+btop)*ext_w+(j+bleft)] = lookat(img, i, j, h, w, ic);
        }
    }
}

/**
 *  @ingroup api_cpu
 *  @brief Extract an image from an extended image
 *
 *  Given an input 2D extended image (with initial conditions) extract
 *  the original image in the middle.
 *
 *  @param[out] img The 2D image to extract
 *  @param[in] w Width of the extracted image
 *  @param[in] h Height of the extracted image
 *  @param[in,out] ext_img The extended 2D image (to be deallocated)
 *  @param[in] ext_w Width of the extended image
 *  @param[in] extb Extension (in blocks) considered in extended image
 *  @tparam T Image value type
 */
template< class T >
void extract_image( T *img,
                    const int& w,
                    const int& h,
                    T *& ext_img,
                    const int& ext_w,
                    const int& extb ) {
    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, w, h, extb );
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            img[i*w+j] = ext_img[(i+btop)*ext_w+(j+bleft)];
        }
    }
    delete [] ext_img;
}

//-- First-order Filter -------------------------------------------------------

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering on columns forward and reverse
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition can be zero, clamp,
 *  repeat or mirror.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void rcfr( T *inout,
           const int& w,
           const int& h,
           const T& b0,
           const T& a1,
           const bool& ff = false ) {
    for (int j = 0; j < w; j++) {
        int i;
        // p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T p = (T)0;
        T c;
        for (i = 0; i < h; i++) {
            c = inout[i*w+j];
            p = c*b0 - p*a1;
            inout[i*w+j] = p;
        }
        if (ff) continue;
        p = (T)0;
        for (i--; i >= 0; i--) {
            c = inout[i*w+j];
            p = c*b0 - p*a1;
            inout[i*w+j] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering on rows forward and reverse
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and a feedback coefficient, i.e. a weight on the
 *  previous element.  The initial condition can be zero, clamp,
 *  repeat or mirror.  The computation is done sequentially in a naïve
 *  single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void rrfr( T *inout,
           const int& w,
           const int& h,
           const T& b0,
           const T& a1,
           const bool& ff = false ) {
    for (int i = 0; i < h; i++, inout += w) {
        int j;
        // p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T p = (T)0;
        T c;
        for (j = 0; j < w; j++) {
            c = inout[j];
            p = c*b0 - p*a1;
            inout[j] = p;
        }
        if (ff) continue;
        p = (T)0;
        for (j--; j >= 0; j--) {
            c = inout[j];
            p = c*b0 - p*a1;
            inout[j] = p;
        }
    }
}
/**
 *  @example example_r1.cc
 *
 *  This is an example of how to use the rrfr() function in the CPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup cpu
 *  @brief Compute first-order recursive filtering
 *
 *  Given an input 2D image compute a first-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and a feedback coefficient, i.e. a weight
 *  on the previous element.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 *  @param[in] ic Initial condition (for outside access) (default zero)
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& w,
        const int& h,
        const T& b0,
        const T& a1,
        const bool& ff = false,
        const int& extb = 0,
        const initcond& ic = zero ) {
    if (extend(w, h, extb)) {
        int ext_w, ext_h;
        float *ext_inout;
        extend_image(ext_inout, ext_w, ext_h, inout, w, h, extb, ic);
        rcfr(ext_inout, ext_w, ext_h, b0, a1, ff);
        rrfr(ext_inout, ext_w, ext_h, b0, a1, ff);
        extract_image(inout, w, h, ext_inout, ext_w, extb);
    } else {
        rcfr(inout, w, h, b0, a1, ff);
        rrfr(inout, w, h, b0, a1, ff);
    }
}

//-- Second-order Filter ------------------------------------------------------

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering on columns forward and reverse
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns with a causal-anticausal filter pair.  The filter
 *  is computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void rcfr( T *inout,
           const int& w,
           const int& h,
           const T& b0,
           const T& a1,
           const T& a2,
           const bool& ff = false ) {
    for (int j = 0; j < w; j++) {
        int i;
        // pp=p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T pp = (T)0;
        T p = (T)0;
        T c;
        for (i = 0; i < h; i++) {
            c = inout[i*w+j];
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            inout[i*w+j] = p;
        }
        if (ff) continue;
        pp = p = (T)0;
        for (i--; i >= 0; i--) {
            c = inout[i*w+j];
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            inout[i*w+j] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering on rows forward and reverse
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its rows with a causal-anticausal filter pair.  The filter is
 *  computed using a feedforward coefficient, i.e. a weight on the
 *  current element, and two feedback coefficients, i.e. weights on
 *  the previous two elements.  The initial condition can be zero,
 *  clamp, repeat or mirror.  The computation is done sequentially in
 *  a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @tparam T Image value type
 */
template< class T >
void rrfr( T *inout,
           const int& w,
           const int& h,
           const T& b0,
           const T& a1,
           const T& a2,
           const bool& ff = false ) {
    for (int i = 0; i < h; i++, inout += w) {
        int j;
        // pp=p=0 is not initial condition based, it is due
        // to the filter order outside image + extension
        T pp = (T)0;
        T p = (T)0;
        T c;
        for (j = 0; j < w; j++) {
            c = inout[j];
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            inout[j] = p;
        }
        if (ff) continue;
        pp = p = (T)0;
        for (j--; j >= 0; j--) {
            c = inout[j];
            c = c*b0 - p*a1 - pp*a2;
            pp = p;
            p = c;
            inout[j] = p;
        }
    }
}

/**
 *  @ingroup cpu
 *  @brief Compute second-order recursive filtering
 *
 *  Given an input 2D image compute a second-order recursive filtering
 *  on its columns and rows with a causal-anticausal filter pair.  The
 *  filter is computed using a feedforward coefficient, i.e. a weight
 *  on the current element, and two feedback coefficients,
 *  i.e. weights on the previous two elements.  The initial condition
 *  can be zero, clamp, repeat or mirror.  The computation is done
 *  sequentially in a naïve single-core CPU fashion.
 *
 *  @param[in,out] inout The 2D image to compute recursive filtering
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 *  @param[in] ff Forward-only (ignore anticausal filter) flag
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 *  @param[in] ic Initial condition (for outside access) (default zero)
 *  @tparam T Image value type
 */
template< class T >
void r( T *inout,
        const int& w,
        const int& h,
        const T& b0,
        const T& a1,
        const T& a2,
        const bool& ff = false,
        const int& extb = 0,
        const initcond& ic = zero ) {
    if (extend(w, h, extb)) {
        int ext_w, ext_h;
        float *ext_inout;
        extend_image(ext_inout, ext_w, ext_h, inout, w, h, extb, ic);
        rcfr(ext_inout, ext_w, ext_h, b0, a1, a2, ff);
        rrfr(ext_inout, ext_w, ext_h, b0, a1, a2, ff);
        extract_image(inout, w, h, ext_inout, ext_w, extb);
    } else {
        rcfr(inout, w, h, b0, a1, a2, ff);
        rrfr(inout, w, h, b0, a1, a2, ff);
    }
}

//-- SAT ----------------------------------------------------------------------

/**
 *  @ingroup api_cpu
 *  @brief Compute the Summed-area Table of an image in the CPU
 *
 *  Given an input 2D image compute its Summed-Area Table (SAT) by
 *  applying a first-order recursive filters forward using zero-border
 *  initial conditions.
 *
 *  @param[in,out] in The 2D image to compute the SAT
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @tparam T Image value type
 */
template< class T >
void sat_cpu( T *in,
              const int& w,
              const int& h ) {
    r(in, w, h, (T)1, (T)-1, true);
}
/**
 *  @example example_sat1.cc
 *
 *  This is an example of how to use the sat_cpu() function in the
 *  CPU.
 *
 *  @see cpuground.h
 */

//-- Gaussian -----------------------------------------------------------------

/**
 *  @ingroup api_cpu
 *  @brief Gaussian blur an image in the CPU
 *
 *  Given an input 2D image compute the Gaussian blur of it by
 *  applying a sequence of recursive filters using clamp-to-border
 *  initial conditions.
 *
 *  @param[in,out] in The 2D image to compute Gaussian blur
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1)
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @tparam T Image value type
 */
template< class T >
void gaussian_cpu( T **in,
                   const int& w,
                   const int& h,
                   const int& depth,
                   const T& s,
                   const int& extb = 1,
                   const initcond& ic = clamp ) {
    T b10, a11, b20, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    for (int c = 0; c < depth; c++) {
        r(in[c], w, h, b10, a11, false, extb, ic);
        r(in[c], w, h, b20, a21, a22, false, extb, ic);
    }
}
/**
 *  @example app_recursive.cc
 *
 *  This is an application example of how to use the gaussian_cpu()
 *  function and bspline3i_cpu() function in the CPU; as well as the
 *  gaussian_gpu() function and bspline3i_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

/**
 *  @ingroup api_cpu
 *  @overload
 *  @brief Gaussian blur a single-channel image in the CPU
 *
 *  @param[in,out] in The single-channel 2D image to compute Gaussian blur
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] s Sigma support of Gaussian blur computation
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1)
 *  @param[in] ic Initial condition (for outside access) (default clamp)
 *  @tparam T Image value type
 */
template< class T >
void gaussian_cpu( T *in,
                   const int& w,
                   const int& h,
                   const T& s,
                   const int& extb = 1,
                   const initcond& ic = clamp ) {
    T b10, a11, b20, a21, a22;
    weights1(s, b10, a11);
    weights2(s, b20, a21, a22);
    r(in, w, h, b10, a11, false, extb, ic);
    r(in, w, h, b20, a21, a22, false, extb, ic);
}
/**
 *  @example example_gauss.cc
 *
 *  This is an example of how to use the gaussian_cpu() function in
 *  the CPU and the gaussian_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

//-- BSpline ------------------------------------------------------------------

/**
 *  @ingroup api_cpu
 *  @brief Compute the Bicubic B-Spline interpolation of an image in the CPU
 *
 *  Given an input 2D image compute the Bicubic B-Spline interpolation
 *  of it by applying a first-order recursive filter using
 *  clamp-to-border initial conditions.
 *
 *  @param[in,out] in The 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] depth Depth of the input image (color channels)
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1)
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @tparam T Image value type
 */
template< class T >
void bspline3i_cpu( T **in,
                    const int& w,
                    const int& h,
                    const int& depth,
                    const int& extb = 1,
                    const initcond& ic = mirror ) {
    const T alpha = (T)2 - sqrt((T)3);
    for (int c = 0; c < depth; c++) {
        r(in[c], w, h, (T)1+alpha, alpha, false, extb, ic);
    }
}

/**
 *  @ingroup api_cpu
 *  @overload
 *  @brief Compute the Bicubic B-Spline interpolation of a single-channel image in the CPU
 *
 *  @param[in,out] in The single-channel 2D image to compute the Bicubic B-Spline interpolation
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] extb Extension (in blocks) to consider outside image (default 1)
 *  @param[in] ic Initial condition (for outside access) (default mirror)
 *  @tparam T Image value type
 */
template< class T >
void bspline3i_cpu( T *in,
                    const int& w,
                    const int& h,
                    const int& extb = 1,
                    const initcond& ic = mirror ) {
    const T alpha = (T)2 - sqrt((T)3);
    r(in, w, h, (T)1+alpha, alpha, false, extb, ic);
}
/**
 *  @example example_bspline.cc
 *
 *  This is an example of how to use the bspline3i_cpu() function in
 *  the CPU and the bspline3i_gpu() function in the GPU.
 *
 *  @see cpuground.h
 */

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // CPUFILTER_H
//=============================================================================
