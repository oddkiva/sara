/**
 *  GPU constants
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUCONSTS_CUH
#define GPUCONSTS_CUH

//== INCLUDES =================================================================

#include <util.h>
#include <gpudefs.h>
#include <symbol.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== DEFINITIONS ===============================================================

__device__
__constant__ int c_width, c_height,
    c_m_size, c_n_size, c_last_m, c_last_n,
    c_border, c_carry_width, c_carry_height;

__device__
__constant__ float c_inv_width, c_inv_height,
    c_b0, c_a1, c_a2, c_inv_b0,
    c_AbF, c_AbR, c_HARB_AFP;

__device__
__constant__ Vector<float,WS> c_TAFB, c_HARB_AFB,
    c_ARE, c_ARB_AFP_T;

__device__
__constant__ Matrix<float,2,2> c_AbF2, c_AbR2,
    c_AFP_HARB;

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup api_gpu
 *  @brief Upload algorithm setup values
 *
 *  Given the algorithm setup, upload the values to the device
 *  constant memory.
 *
 *  @param[in] algs Algorithm setup to upload to the GPU
 */
__host__
void up_alg_setup( const alg_setup& algs ) {

	copy_to_symbol(c_width, algs.width);
    copy_to_symbol(c_height, algs.height);
    copy_to_symbol(c_m_size, algs.m_size);
	copy_to_symbol(c_n_size, algs.n_size);
    copy_to_symbol(c_last_m, algs.last_m);
    copy_to_symbol(c_last_n, algs.last_n);
    copy_to_symbol(c_border, algs.border);
	copy_to_symbol(c_carry_width, algs.carry_width);
    copy_to_symbol(c_carry_height, algs.carry_height);
    copy_to_symbol(c_inv_width, algs.inv_width);
    copy_to_symbol(c_inv_height, algs.inv_height);

}

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants first-order coefficients
 *
 *  Given the first-order coefficients of the recursive filter, upload
 *  to the device constant memory the coefficients-related values.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 */
__host__
void up_constants_coefficients1( const float& b0,
                                 const float& a1 ) {

    copy_to_symbol(c_b0, b0);
    copy_to_symbol(c_a1, a1);
    copy_to_symbol(c_inv_b0, 1.f/b0);

    const int B = WS, R = 1;

    Vector<float,R+1> w;
    w[0] = b0;
    w[1] = a1;

    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w),
                      ARB_T = rev(Ib, Zbr, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T),
                      AbF = transp(AbF_T),
                      AbR = transp(AbR_T),
                      HARB_AFP_T = AFP_T*head<R>(ARB_T),
                      HARB_AFP = transp(HARB_AFP_T);
    Matrix<float,R,B> ARB_AFP_T = AFP_T*ARB_T,
                      TAFB = transp(tail<R>(AFB_T)),
                      HARB_AFB = transp(AFB_T*head<R>(ARB_T));

    copy_to_symbol(c_AbF, AbF[0][0]);
    copy_to_symbol(c_AbR, AbR[0][0]);
    copy_to_symbol(c_HARB_AFP, HARB_AFP[0][0]);

    copy_to_symbol(c_ARE, ARE_T[0]);
    copy_to_symbol(c_ARB_AFP_T, ARB_AFP_T[0]);
    copy_to_symbol(c_TAFB, TAFB[0]);
    copy_to_symbol(c_HARB_AFB, HARB_AFB[0]);

}

/**
 *  @ingroup api_gpu
 *  @brief Upload device constants second-order coefficients
 *
 *  Given the second-order coefficients of the recursive filter,
 *  upload to the device constant memory the coefficients-related
 *  values.
 *
 *  @param[in] b0 Feedforward coefficient
 *  @param[in] a1 Feedback first-order coefficient
 *  @param[in] a2 Feedback second-order coefficient
 */
__host__
void up_constants_coefficients2( const float& b0,
                                 const float& a1,
                                 const float& a2 ) {

    copy_to_symbol(c_b0, b0);
    copy_to_symbol(c_a1, a1);
    copy_to_symbol(c_a2, a2);
    copy_to_symbol(c_inv_b0, 1.f/b0);

    const int B = WS, R = 2;

    Vector<float,R+1> w;
    w[0] = b0;
    w[1] = a1;
    w[2] = a2;

    Matrix<float,R,R> Ir = identity<float,R,R>();
    Matrix<float,B,R> Zbr = zeros<float,B,R>();
    Matrix<float,R,B> Zrb = zeros<float,R,B>();
    Matrix<float,B,B> Ib = identity<float,B,B>();

    Matrix<float,R,B> AFP_T = fwd(Ir, Zrb, w),
                      ARE_T = rev(Zrb, Ir, w);
    Matrix<float,B,B> AFB_T = fwd(Zbr, Ib, w),
                      ARB_T = rev(Ib, Zbr, w);

    Matrix<float,R,R> AbF_T = tail<R>(AFP_T),
                      AbR_T = head<R>(ARE_T),
                      HARB_AFP_T = AFP_T*head<R>(ARB_T);

    copy_to_symbol(c_AbF2, AbF_T);
    copy_to_symbol(c_AbR2, AbR_T);
    copy_to_symbol(c_AFP_HARB, HARB_AFP_T);

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUCONSTS_CUH
//=============================================================================
