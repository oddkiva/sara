/**
 *  @file alg5.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 5
 *  @author Rodolfo Lima
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <cmath>
#include <cstdio>
#include <cfloat>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <dvector.h>
#include <extension.h>

#include <gpufilter.h>
#include <gpuconsts.cuh>

#include <alg5.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Algorithm 5_1 Stage 1 ----------------------------------------------------

__global__ __launch_bounds__(WS*DW, ONB)
void alg5_stage1( float *g_transp_pybar,
                  float *g_transp_ezhat,
                  float *g_ptucheck,
                  float *g_etvtilde )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x*2, n = blockIdx.y;

    // Each cuda block will work on two horizontally adjacent WSxWS
    // input data blocks, so allocate enough shared memory for these.
    __shared__ float s_block[WS*2][WS+1];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx],
          (*bdata2)[WS+1] = (float (*)[WS+1])&s_block[ty+WS][tx];

    // Load data into shared memory
    float tu = ((m-c_border)*WS+tx+.5f)*c_inv_width,
        tv = ((n-c_border)*WS+ty+.5f)*c_inv_height;

    int i;
#pragma unroll
    for (i=0; i<WS-(WS%DW); i+=DW)
    {
        **bdata = tex2D(t_in, tu, tv);
        bdata += DW;

        **bdata2 = tex2D(t_in, tu+WS*c_inv_width, tv);
        bdata2 += DW;

        tv += DW*c_inv_height;
    }

    if (ty < WS%DW)
    {
        **bdata = tex2D(t_in, tu, tv);
        **bdata2 = tex2D(t_in, tu+WS*c_inv_width, tv);
    }

    m += ty;

    // We use a transposed matrix for pybar and ezhat to have
    // coalesced memory accesses. This is the index for these
    // transposed buffers.
    g_transp_pybar += m*c_carry_height + n*WS + tx; 
    g_transp_ezhat += m*c_carry_height + n*WS + tx;
    g_ptucheck += n*c_carry_width + m*WS + tx;
    g_etvtilde += n*c_carry_width + m*WS + tx;

    __syncthreads();

    if (m >= c_m_size)
        return;

    float prev;

    if (ty < 2)
    {
        { // scan rows
            float *bdata = s_block[tx+ty*WS];

            // calculate pybar, scan left -> right
            prev = *bdata++;

#pragma unroll
            for (int j=1; j<WS; ++j, ++bdata)
                prev = *bdata -= prev*c_a1;

            *g_transp_pybar = prev*c_b0;

            // calculate ezhat, scan right -> left
            prev = *--bdata;
            --bdata;

#pragma unroll
            for (int j=WS-2; j>=0; --j, --bdata)
                prev = *bdata -= prev*c_a1;

            *g_transp_ezhat = prev*(c_b0*c_b0);
        }

        { // scan columns
            // ty*WS makes this warp's bdata point to the right data
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty*WS][tx];

            // calculate ptucheck, scan top -> down
            prev = **bdata++;

#pragma unroll
            for (int i=1; i<WS; ++i, ++bdata)
                prev = **bdata -= prev*c_a1;

            *g_ptucheck = prev*c_b0*c_b0*c_b0;

            // calculate etvtilde, scan bottom -> up
            if (n > 0)
            {
                prev = **--bdata;
                --bdata;

#pragma unroll
                for (int i=WS-2; i>=0; --i, --bdata)
                    prev = **bdata - prev*c_a1;

                *g_etvtilde = prev*c_b0*c_b0*c_b0*c_b0;
            }
        }
    }
}

//-- Algorithm 5_1 Stage 2 and 3 ----------------------------------------------

__global__ __launch_bounds__(WS*DW, DNB)
void alg5_stage2_3( float *g_transp_pybar,
                    float *g_transp_ezhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ float s_transp_block[DW][WS];
    float *bdata = &s_transp_block[ty][tx];

    // P(ybar) -> P(y) processing --------------------------------------
    if (c_m_size<=1)
        return;

    float *transp_pybar = g_transp_pybar + ty*c_carry_height + n*WS+tx;

    // first column-block

    // read P(ybar)
    *bdata = *transp_pybar;

    float py; // P(Y)

    __syncthreads();

    if (ty == 0)
    {
        float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];

        // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
        py = **bdata++;

#pragma unroll
        for (int m=1; m<blockDim.y; ++m, ++bdata)
            **bdata = py = **bdata + c_AbF*py;
    }

    __syncthreads();

    // write P(y)
    if (ty > 0) // first one doesn't need fixing
        *transp_pybar = *bdata;

    transp_pybar += c_carry_height*blockDim.y;

    // middle column-blocks
    int m = blockDim.y;
    if (m == DW)
    {
        int mmax = c_m_size-(c_m_size%DW)-1;
        for (; m<mmax; m+=DW)
        {
            *bdata = *transp_pybar;

            __syncthreads();

            if (ty == 0)
            {
                float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
                for (int dm=0; dm<DW; ++dm, ++bdata)
                    **bdata = py = **bdata + c_AbF*py;
            }

            __syncthreads();

            *transp_pybar = *bdata;
            transp_pybar += c_carry_height*DW;
        }
    }

    // remaining column-blocks
    if (m < c_m_size-1)
    {
        if (m+ty < c_m_size-1)
            *bdata = *transp_pybar;

        int remaining = c_m_size-1 - m;

        __syncthreads();

        if (ty == 0)
        {
            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
            for (int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = py = **bdata + c_AbF*py;

        }

        __syncthreads();

        if (m+ty < c_m_size-1)
            *transp_pybar = *bdata;
    }

    // E(zhat) -> E(z) processing --------------------------------------
    int idx = (c_m_size-1-ty)*c_carry_height + n*WS+tx;

    const float *transp_pm1y = g_transp_pybar + idx - c_carry_height;

    // last column-block
    float *transp_ezhat = g_transp_ezhat + idx;

    m = c_m_size-1;

    // all pybars must be updated!
    __syncthreads();

    float ez;

    {
        *bdata = *transp_ezhat;

        if (m-ty > 0)
            *bdata += *transp_pm1y*c_HARB_AFP;

        __syncthreads();

        if (ty == 0)
        {
            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
            ez = **bdata++;

            for (int dm=1; dm<blockDim.y; ++dm, ++bdata)
                **bdata = ez = **bdata + c_AbR*ez;
        }

        __syncthreads();

        *transp_ezhat = *bdata;

    }
    transp_ezhat -= c_carry_height*blockDim.y;
    transp_pm1y -= c_carry_height*blockDim.y;

    // middle column-blocks
    m = c_m_size-1 - blockDim.y;
    if (blockDim.y == DW)
    {
        int mmin = c_m_size%DW;
        for (; m>=mmin; m-=DW)
        {
            if (m > 0)
            {
                *bdata = *transp_ezhat;

                if (m-ty > 0)
                    *bdata += *transp_pm1y*c_HARB_AFP;

                __syncthreads();

                if (ty == 0)
                {
                    float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
                    for (int dm=0; dm<DW; ++dm, ++bdata)
                        **bdata = ez = **bdata + c_AbR*ez;
                }

                __syncthreads();

                *transp_ezhat = *bdata;
            }

            transp_ezhat -= DW*c_carry_height;
            transp_pm1y -= DW*c_carry_height;
        }
    }

    // remaining column-blocks (except first column-block, it isn't needed)
    if (m > 0)
    {
        if (m-ty > 0)
        {
            *bdata = *transp_ezhat;
        
            if (m-ty > 0)
                *bdata += *transp_pm1y*c_HARB_AFP;
        }

        __syncthreads();

        if (ty == 0)
        {
            int remaining = m;

            float (*bdata)[WS] = (float (*)[WS]) &s_transp_block[0][tx];
            // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
#pragma unroll
            for (int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = ez = **bdata + c_AbR*ez;
        }

        __syncthreads();

        if (m-ty > 0)
            *transp_ezhat = *bdata;
    }
}

//-- Algorithm 5_1 Stage 4 and 5 ----------------------------------------------

__global__ __launch_bounds__(WS*CHW, ONB)
void alg5_stage4_5( float *g_ptucheck,
                    float *g_etvtilde,
                    const float *g_transp_py,
                    const float *g_transp_ez )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x;

    __shared__ float s_block[CHW][WS];
    float *bdata = &s_block[ty][tx];

    // P(ucheck) -> P(u) processing --------------------------------------

	volatile __shared__ float s_block_RD_raw[CHW][WS/2+WS+1];
	volatile float (*block_RD)[WS/2+WS+1] = 
            (float (*)[WS/2+WS+1]) &s_block_RD_raw[0][WS/2];

    if (ty < CHW)
        s_block_RD_raw[ty][tx] = 0;

#define CALC_DOT(RES, V1, V2, last) \
    block_RD[ty][tx] = V1*V2; \
    block_RD[ty][tx] += block_RD[ty][tx-1]; \
    block_RD[ty][tx] += block_RD[ty][tx-2]; \
    block_RD[ty][tx] += block_RD[ty][tx-4]; \
    block_RD[ty][tx] += block_RD[ty][tx-8]; \
    block_RD[ty][tx] += block_RD[ty][tx-16]; \
    float RES = block_RD[ty][last];

    {
    float *ptucheck = g_ptucheck + m*WS+tx + ty*c_carry_width;

    // first row-block
    int idx = m*c_carry_height + ty*WS+tx;

    const float *transp_pm1y = g_transp_py + idx - c_carry_height,
                *transp_em1z = g_transp_ez + idx + c_carry_height;

    float ptu;

    if (ty < c_n_size-1)
    {
        // read P(ucheck)
        *bdata = *ptucheck;

        if (m < c_m_size-1)
        {
            CALC_DOT(dot, *transp_em1z, c_TAFB[tx], WS-1);
            *bdata += dot*c_ARE[tx];
        }

        if (m > 0)
        {
            CALC_DOT(dot, *transp_pm1y, c_TAFB[tx], WS-1);
            *bdata += dot*c_ARB_AFP_T[tx];
        }

        transp_pm1y += WS*blockDim.y;
        transp_em1z += WS*blockDim.y;

        __syncthreads();

        if (ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) bdata;

            ptu = **bdata2++;

#pragma unroll
            for (int n=1; n<blockDim.y; ++n, ++bdata2)
                **bdata2 = ptu = **bdata2 + c_AbF*ptu;
        }

        __syncthreads();

        // write P(u)
        *ptucheck = *bdata;

    }
    ptucheck += blockDim.y*c_carry_width;

    // middle row-blocks
    int n = blockDim.y;
    if (n == CHW)
    {
        int nmax = c_n_size-(c_n_size%CHW);

        for (; n<nmax; n+=CHW)
        {
            if (n < c_n_size-1)
            {
                *bdata = *ptucheck;

                if (m < c_m_size-1)
                {
                    CALC_DOT(dot, *transp_em1z, c_TAFB[tx], WS-1);
                    *bdata += dot*c_ARE[tx];

                }

                if (m > 0)
                {
                    CALC_DOT(dot, *transp_pm1y, c_TAFB[tx], WS-1);
                    *bdata += dot*c_ARB_AFP_T[tx];
                }

                transp_pm1y += WS*CHW;
                transp_em1z += WS*CHW;

                __syncthreads();

                if (ty == 0)
                {
                    float (*bdata2)[WS] = (float (*)[WS]) bdata;

#pragma unroll
                    for (int dn=0; dn<CHW; ++dn, ++bdata2)
                        **bdata2 = ptu = **bdata2 + c_AbF*ptu;
                }

                __syncthreads();

                *ptucheck = *bdata;
            }

            ptucheck += CHW*c_carry_width;
        }
    }

    // remaining row-blocks
    if (n < c_n_size-1)
    {
        if (n+ty < c_n_size-1)
        {
            *bdata = *ptucheck;

            if (m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1z, c_TAFB[tx], WS-1);
                *bdata += dot*c_ARE[tx];
            }

            if (m > 0)
            {
                CALC_DOT(dot, *transp_pm1y, c_TAFB[tx], WS-1);
                *bdata += dot*c_ARB_AFP_T[tx];
            }
        }

        __syncthreads();

        if (ty == 0)
        {
            int remaining = c_n_size-1-n;

            float (*bdata2)[WS] = (float (*)[WS]) bdata;
#pragma unroll
            for (int dn=0; dn<remaining; ++dn, ++bdata2)
                **bdata2 = ptu = **bdata2 + c_AbF*ptu;
        }

        __syncthreads();

        if (n+ty < c_n_size-1)
            *ptucheck = *bdata;
    }
    }

    // E(utilde) -> E(u) processing --------------------------------------

    // last row-block
    int idx = (c_n_size-1-ty)*c_carry_width + m*WS+tx;

    float *etvtilde = g_etvtilde + idx;
    const float *ptmn1u = g_ptucheck + idx - c_carry_width;

    int transp_idx = m*c_carry_height + (c_n_size-1-ty)*WS+tx;
    const float *transp_pm1y = g_transp_py + transp_idx-c_carry_height;
    const float *transp_em1z = g_transp_ez + transp_idx+c_carry_height;

    // all ptuchecks must be updated!
    __syncthreads();

    float etv;

    int n = c_n_size-1 - ty;

    {
        *bdata = *etvtilde;
        
        if (m < c_m_size-1)
        {
            CALC_DOT(dot, *transp_em1z, c_HARB_AFB[tx], WS-1);
            *bdata += dot*c_ARE[tx];
        }

        if (m > 0)
        {
            CALC_DOT(dot, *transp_pm1y, c_HARB_AFB[tx], WS-1);

            *bdata += dot*c_ARB_AFP_T[tx];
        }

        if (n > 0)
            *bdata += *ptmn1u*c_HARB_AFP;

        transp_pm1y -= WS*blockDim.y;
        transp_em1z -= WS*blockDim.y;
        ptmn1u -= c_carry_width*blockDim.y;

        __syncthreads();

        if (ty == 0)
        {
            float (*bdata2)[WS] = (float (*)[WS]) bdata;

            etv = **bdata2++;

#pragma unroll
            for (int dn=1; dn<blockDim.y; ++dn, ++bdata2)
                **bdata2 = etv = **bdata2 + c_AbR*etv;
        }

        __syncthreads();

        *etvtilde = *bdata;

        etvtilde -= c_carry_width*blockDim.y;

        n -= blockDim.y;
    }

    // middle row-blocks
    if (blockDim.y == CHW)
    {
        int nmin = c_n_size%CHW;

        for (; n>=nmin; n-=CHW)
        {
            *bdata = *etvtilde;

            if (m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1z, c_HARB_AFB[tx], WS-1);
                *bdata += dot*c_ARE[tx];
            }

            if (m > 0)
            {
                CALC_DOT(dot, *transp_pm1y, c_HARB_AFB[tx], WS-1);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

            if (n > 0)
                *bdata += *ptmn1u*c_HARB_AFP;

            transp_pm1y -= WS*CHW;
            transp_em1z -= WS*CHW;
            ptmn1u -= CHW*c_carry_width;

            __syncthreads();

            if (ty == 0)
            {
                float (*bdata2)[WS] = (float (*)[WS]) bdata;
#pragma unroll
                for (int dn=0; dn<CHW; ++dn, ++bdata2)
                    **bdata2 = etv = **bdata2 + c_AbR*etv;
            }

            __syncthreads();

            *etvtilde = *bdata;

            etvtilde -= CHW*c_carry_width;
        }
    }

    // remaining row-blocks
    if (n+ty >= 0)
    {
        if (n > 0)
        {
            *bdata = *etvtilde + *ptmn1u*c_HARB_AFP;

            if (m < c_m_size-1)
            {
                CALC_DOT(dot, *transp_em1z, c_HARB_AFB[tx], WS-1);
                *bdata += dot*c_ARE[tx];
            }

            if (m > 0)
            {
                CALC_DOT(dot, *transp_pm1y, c_HARB_AFB[tx], WS-1);
                *bdata += dot*c_ARB_AFP_T[tx];
            }

        }

        __syncthreads();

        if (ty == 0)
        {
            int remaining = n+1;
            float (*bdata2)[WS] = (float (*)[WS]) bdata;
#pragma unroll
            for (int dn=0; dn<remaining; ++dn, ++bdata2)
                **bdata2 = etv = **bdata2 + c_AbR*etv;
        }

        __syncthreads();

        if (n > 0)
            *etvtilde = *bdata;
    }
#undef CALC_DOT
}

//-- Algorithm 5_1 Stage 6 ----------------------------------------------------

__global__ __launch_bounds__(WS*DW, CHB)
void alg5_stage6( float *g_out,
                  const float *g_transp_py,
                  const float *g_transp_ez,
                  const float *g_ptu,
                  const float *g_etv )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x*2, n = blockIdx.y;

    __shared__ float s_block[2*WS][WS+1];

    __shared__ float s_py[2][WS], s_ez[2][WS], 
        s_ptu[2][WS], s_etv[2][WS];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx],
          (*bdata2)[WS+1] = (float (*)[WS+1])&s_block[ty+WS][tx];

    bool inside = m+1 >= c_border && m <= c_last_m &&
                    n >= c_border && n <= c_last_n;

    if (inside)
    {
        {
            // load data into shared memory
            float tu = ((m-c_border)*WS+tx + 0.5f)*c_inv_width,
                  tv = ((n-c_border)*WS+ty + 0.5f)*c_inv_height;

#pragma unroll
            for (int i=0; i<WS-(WS%DW); i+=DW)
            {
                **bdata = tex2D(t_in, tu, tv);
                bdata += DW;

                **bdata2 = tex2D(t_in, tu+WS*c_inv_width, tv);
                bdata2 += DW;

                tv += DW*c_inv_height;
            }

            if (ty < WS%DW)
            {
                **bdata = tex2D(t_in, tu, tv);
                **bdata2 = tex2D(t_in, tu+WS*c_inv_width, tv);
            }
        }

        if (ty < 2)
        {
            m += ty;

            if (m >= c_border && m <= c_last_m)
            {
                if (m > 0)
                    s_py[ty][tx] = g_transp_py[(n*WS + tx) + (m-1)*c_carry_height] * c_inv_b0;
                else
                    s_py[ty][tx] = 0;
            }
        }
        else if (ty < 4)
        {
            m += ty-2;

            if (m >= c_border && m <= c_last_m)
            {
                if (m < c_m_size-1)
                    s_ez[ty-2][tx] = g_transp_ez[(n*WS + tx) + (m+1)*c_carry_height];
                else
                    s_ez[ty-2][tx] = 0;
            }
        }
        else if (ty < 6)
        {
            m += ty-4;

            if (m >= c_border && m <= c_last_m)
            {
                if (n > 0)
                    s_ptu[ty-4][tx] = g_ptu[(m*WS + tx) + (n-1)*c_carry_width] * c_inv_b0;
                else
                    s_ptu[ty-4][tx] = 0;
            }
        }
        else if (ty < 8)
        {
            m += ty-6;

            if (m >= c_border && m <= c_last_m)
            {
                if (n < c_n_size-1)
                    s_etv[ty-6][tx] = g_etv[(m*WS + tx) + (n+1)*c_carry_width];
                else
                    s_etv[ty-6][tx] = 0;
            }
        }
    }

    __syncthreads();

    if (!inside || m < c_border || m > c_last_m)
        return;

    if (ty < 2)
    {
        const float b0_2 = c_b0*c_b0;

        // scan rows
        {
            float *bdata = s_block[tx+ty*WS];

            // calculate y ---------------------

            float prev = s_py[ty][tx];

#pragma unroll
            for (int j=0; j<WS; ++j, ++bdata)
                *bdata = prev = *bdata - prev*c_a1;

            // calculate z ---------------------

            prev = s_ez[ty][tx];
            --bdata;

#pragma unroll
            for (int j=WS-1; j>=0; --j, --bdata)
                *bdata = prev = *bdata*b0_2 - prev*c_a1;
        }

        // scan columns
        {
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty*WS][tx];

            // calculate u ---------------------

            float prev = s_ptu[ty][tx];

#pragma unroll
            for (int i=0; i<WS; ++i, ++bdata)
                **bdata = prev = **bdata - prev*c_a1;

            // calculate v ---------------------

            int x = (m-c_border)*WS+tx;
            if (x >= c_width)
                return;

            prev = s_etv[ty][tx];
            --bdata;

            int y = (n-c_border+1)*WS-1;

            if (y >= c_height)
            {
                int i;

#pragma unroll
                for (i=y; i>=c_height; --i)
                     prev = **bdata-- *b0_2 - prev*c_a1;

                float *out = g_out + (c_height-1)*c_width + x;

#pragma unroll
                for (;i>=(n-c_border)*WS; --i)
                {
                    *out = prev = **bdata-- *b0_2 - prev*c_a1;
                    out -= c_width;
                }

            }
            else
            {
                float *out = g_out + y*c_width + x;

#pragma unroll
                for (int i=WS-1; i>=0; --i) 
                {
                    *out = prev = **bdata-- *b0_2 - prev*c_a1;
                    out -= c_width;
                }
            }
        }
    }
}

//-- Host ---------------------------------------------------------------------

__host__
void prepare_alg5( alg_setup& algs,
                   dvector<float>& d_out,
                   dvector<float>& d_transp_pybar,
                   dvector<float>& d_transp_ezhat,
                   dvector<float>& d_ptucheck,
                   dvector<float>& d_etvtilde,
                   cudaArray *& a_in,
                   const float *h_in,
                   const int& w,
                   const int& h,
                   const float& b0,
                   const float& a1,
                   const int& extb,
                   const initcond& ic )
{

    up_constants_coefficients1( b0, a1 );

    d_out.resize( w * h );

    calc_alg_setup( algs, w, h, extb );
    up_alg_setup( algs );

    d_transp_pybar.resize( algs.m_size * algs.carry_height );
    d_transp_ezhat.resize( algs.m_size * algs.carry_height );
    d_ptucheck.resize( algs.n_size  * algs.carry_width );
    d_etvtilde.resize( algs.n_size * algs.carry_width );

    d_transp_pybar.fill_zero();
    d_transp_ezhat.fill_zero();
    d_ptucheck.fill_zero();
    d_etvtilde.fill_zero();

    up_texture( a_in, h_in, w, h, ic );

}

__host__
void alg5( dvector<float>& d_out,
           dvector<float>& d_transp_pybar,
           dvector<float>& d_transp_ezhat,
           dvector<float>& d_ptucheck,
           dvector<float>& d_etvtilde,
           const cudaArray *a_in,
           const alg_setup& algs )
{

    dvector<float> d_transp_py, d_transp_ez, d_ptu, d_etv;

    cudaBindTextureToArray( t_in, a_in );

    alg5_stage1<<<
        dim3((algs.m_size+2-1)/2, algs.n_size), dim3(WS, DW) >>>(
            d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde );

    alg5_stage2_3<<<
        dim3(1, algs.n_size), dim3(WS, std::min<int>(algs.m_size, DW)) >>>(
            d_transp_pybar, d_transp_ezhat );

    swap(d_transp_pybar, d_transp_py);
    swap(d_transp_ezhat, d_transp_ez);

    alg5_stage4_5<<<
        dim3(algs.m_size, 1), dim3(WS, std::min<int>(algs.n_size, CHW)) >>>(
            d_ptucheck, d_etvtilde, d_transp_py, d_transp_ez );

    swap(d_ptucheck, d_ptu);
    swap(d_etvtilde, d_etv);

    alg5_stage6<<<
        dim3((algs.m_size+2-1)/2, algs.n_size), dim3(WS, DW) >>>(
            d_out, d_transp_py, d_transp_ez, d_ptu, d_etv );

    swap(d_etv, d_etvtilde);
    swap(d_ptu, d_ptucheck);
    swap(d_transp_ez, d_transp_ezhat);
    swap(d_transp_py, d_transp_pybar);

    cudaUnbindTexture( t_in );

}

__host__
void alg5( float *h_inout,
           const int& w,
           const int& h,
           const float& b0,
           const float& a1,
           const int& extb,
           const initcond& ic )
{

    alg_setup algs;
    dvector<float> d_out;
    dvector<float> d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde;
    cudaArray *a_in;

    prepare_alg5( algs, d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck,
                  d_etvtilde, a_in, h_inout, w, h, b0, a1, extb, ic );

    alg5( d_out, d_transp_pybar, d_transp_ezhat, d_ptucheck, d_etvtilde,
          a_in, algs );

    d_out.copy_to( h_inout, w * h );

    cudaFreeArray( a_in );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
