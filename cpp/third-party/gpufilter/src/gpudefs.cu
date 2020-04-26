/**
 *  GPU definitions (implementation)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

//== INCLUDES =================================================================

#include <vector>
#include <complex>

#include <util.h>
#include <symbol.h>

#include <gpudefs.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h ) {

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+WS-1)/WS;
    algs.n_size = (h+WS-1)/WS;
    algs.last_m = algs.m_size-1;
    algs.last_n = algs.n_size-1;
    algs.border = 0;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.carry_height = h;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}

__host__
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h,
                     const int& extb ) {

    int bleft, btop, bright, bbottom;
    calc_borders( bleft, btop, bright, bbottom, w, h, extb );

    algs.width = w;
    algs.height = h;
    algs.m_size = (w+bleft+bright+WS-1)/WS;
    algs.n_size = (h+btop+bbottom+WS-1)/WS;
    algs.last_m = (bleft+w-1)/WS;
    algs.last_n = (btop+h-1)/WS;
    algs.border = extb;
    algs.carry_width = algs.m_size*WS;
    algs.carry_height = algs.n_size*WS;
    algs.inv_width = 1.f/(float)w;
    algs.inv_height = 1.f/(float)h;

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
