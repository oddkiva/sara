/**
 *  @file alg4.cu
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 4
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

#include <util.h>

#include <gpufilter.h>
#include <gpuconsts.cuh>

#include <alg4.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Utilities ----------------------------------------------------------------

template <class T> 
__device__ inline void swap(T& a, T& b) {
    T c = a;
    a = b;
    b = c;
}

__device__ float2 operator + ( const float2 &a,
                               const float2 &b ) {
    return make_float2(a.x+b.x, a.y+b.y);
}


__device__ float2& operator += ( float2& a,
                                 const float2& b ) {
    a.x += b.x;
    a.y += b.y;
    return a;
}


__device__ float2 operator * ( const float2& a,
                               float b ) {
    return make_float2(a.x*b, a.y*b);
}


__device__ float2 operator * ( float a,
                               const float2& b ) {
    return b*a;
}


__device__ float2 operator / ( const float2& a,
                               float b ) {
    return make_float2(a.x/b, a.y/b);
}


__device__ float2 mul2x2( const float2& v,
                          Matrix<float,2,2> mat) {
    return make_float2(v.x*mat[0][0] + v.y*mat[1][0],
                       v.x*mat[0][1] + v.y*mat[1][1]);
}

//-- Algorithm 4_2 Stage 1 ----------------------------------------------------

__global__ __launch_bounds__(WS*DW, DNB)
void alg4_stage1( float2 *g_transp_pybar,
                  float2 *g_transp_ezhat )
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

    m += ty;

    if (m >= c_m_size)
        return;

    // We use a transposed matrix for pybar and ezhat to have
    // coalesced memory accesses. This is the index for these
    // transposed buffers.
    g_transp_pybar += m*c_carry_height + n*WS + tx; 
    g_transp_ezhat += m*c_carry_height + n*WS + tx;

    __syncthreads();

    float2 prev; // .x -> p0, .y -> p1

    if (ty < 2)
    {
        float *bdata = s_block[tx+ty*WS];

        // calculate pybar, scan left -> right
        prev = make_float2(0,*bdata++);

#pragma unroll
        for (int j=1; j<WS; ++j, ++bdata)
        {
            *bdata = prev.x = *bdata - prev.y*c_a1 - prev.x*c_a2;

            swap(prev.x, prev.y);
        }

        if (m < c_m_size-1)
            *g_transp_pybar = prev*c_b0;


        if (m > 0)
        {
            // calculate ezhat, scan right -> left
            prev = make_float2(*--bdata, 0);

            --bdata;

#pragma unroll
            for (int j=WS-2; j>=0; --j, --bdata)
            {
                *bdata = prev.y = *bdata - prev.x*c_a1 - prev.y*c_a2;
                swap(prev.x, prev.y);
            }

            *g_transp_ezhat = prev*(c_b0*c_b0);
        }
    }
}

//-- Algorithm 4_2 Stage 2 and 3 or Stage 5 and 6 -----------------------------

__device__
void alg4_stage2_3v5_6( float2 *g_transp_pybar,
                        float2 *g_transp_ezhat )
{
    int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.y;

    __shared__ float2 s_transp_block[DW][WS];
    float2 *bdata = &s_transp_block[ty][tx];

    // P(ybar) -> P(y) processing --------------------------------------

    float2 *transp_pybar = g_transp_pybar + ty*c_carry_height + n*WS+tx;

    // first column-block

    // read P(ybar)
    *bdata = *transp_pybar;

    float2 py; // P(Y), .x = p0, .y = p1

    __syncthreads();

    if (ty == 0)
    {
        float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];

        // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
        py = **bdata++;

#pragma unroll
        for (int m=1; m<blockDim.y; ++m, ++bdata)
            **bdata = py = **bdata + mul2x2(py,c_AbF2);
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
                float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
                for (int dm=0; dm<DW; ++dm, ++bdata)
                    **bdata = py = **bdata + mul2x2(py,c_AbF2);
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
            float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
            for (int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = py = **bdata + mul2x2(py,c_AbF2);

        }

        __syncthreads();

        if (m+ty < c_m_size-1)
            *transp_pybar = *bdata;
    }

    // E(zhat) -> E(z) processing --------------------------------------

    int idx = (c_m_size-1-ty)*c_carry_height + n*WS+tx;

    const float2 *transp_pm1y = g_transp_pybar + idx - c_carry_height;

    // last column-block
    float2 *transp_ezhat = g_transp_ezhat + idx;

    m = c_m_size-1;

    // all pybars must be updated!
    __syncthreads();

    float2 ez;

    if (m-ty > 0)
    {
        *bdata = *transp_ezhat;

        *bdata += mul2x2(*transp_pm1y,c_AFP_HARB);

        __syncthreads();

        if (ty == 0)
        {
            float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];
            ez = **bdata++;

            for (int dm=1; dm<blockDim.y; ++dm, ++bdata)
                **bdata = ez = **bdata + mul2x2(ez,c_AbR2);
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
                    *bdata += mul2x2(*transp_pm1y,c_AFP_HARB);

                __syncthreads();

                if (ty == 0)
                {
                    float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];
#pragma unroll
                    for (int dm=0; dm<DW; ++dm, ++bdata)
                        **bdata = ez = **bdata + mul2x2(ez,c_AbR2);
                }

                __syncthreads();

                *transp_ezhat = *bdata;
            }

            transp_ezhat -= DW*c_carry_height;
            transp_pm1y -= DW*c_carry_height;
        }
    }

    // remaining column-blocks
    if (m > 0)
    {
        int remaining = m+1;

        if (m-ty >= 0)
        {
            *bdata = *transp_ezhat;
        
            if (m-ty > 0)
                *bdata += mul2x2(*transp_pm1y,c_AFP_HARB);
        }

        __syncthreads();

        if (ty == 0)
        {
            float2 (*bdata)[WS] = (float2 (*)[WS]) &s_transp_block[0][tx];
            // (24): P_m(y) = P_m(ybar) + A^b_F * P_{m-1}(y)
#pragma unroll
            for (int dm=0; dm<remaining; ++dm, ++bdata)
                **bdata = ez = **bdata + mul2x2(ez,c_AbR2);
        }

        __syncthreads();

        if (m-ty > 0)
            *transp_ezhat = *bdata;
    }
}

//-- Algorithm 4_2 Stage 4 or Stage 7 -----------------------------------------

template <bool p_fusion>
__device__
void alg4_stage4v7( float *g_transp_out,
                    float2 *g_transp_py,
                    float2 *g_transp_ez,
                    float2 *g_pubar,
                    float2 *g_evhat,
                    int out_stride )
{
    int tx = threadIdx.x, ty = threadIdx.y, m = blockIdx.x*2, n = blockIdx.y;

    // Each cuda block will work on two horizontally adjacent WSxWS
    // input data blocks, so allocate enough shared memory for these.
    __shared__ float s_block[WS*2][WS+1];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx],
          (*bdata2)[WS+1] = (float (*)[WS+1])&s_block[ty+WS][tx];

    // Load data into shared memory
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

    m += ty;

    if (m >= c_m_size)
        return;

    // We use a transposed matrix for py and ez to have coalesced
    // memory accesses. This is the index for these transposed
    // buffers.
    g_transp_py += (m-1)*c_carry_height + n*WS + tx; 
    g_transp_ez += (m+1)*c_carry_height + n*WS + tx;

    __syncthreads();

    if (ty < 2)
    {
        float2 prev; // .x -> p0, .y -> p1

        float *bdata = s_block[tx+ty*WS];

        // calculate pybar, scan left -> right
        if (m > 0)
            prev = *g_transp_py * c_inv_b0;
        else
            prev = make_float2(0,0);

#pragma unroll
        for (int j=0; j<WS; ++j, ++bdata)
        {
            *bdata = prev.x = *bdata - prev.y*c_a1 - prev.x*c_a2;

            swap(prev.x, prev.y);
        }
        --bdata;

        // calculate ez, scan right -> left
        if (m < c_m_size-1)
            prev = *g_transp_ez;
        else
            prev = make_float2(0,0);

        float b0_2 = c_b0*c_b0;

        // For some reason it's faster when this is here then inside
        // the next if block
        int x = (m-c_border+1)*WS-1;
        int y = (n-c_border)*WS+tx;

        // current block intersects transp_out's area?
        if (m >= c_border && m <= c_last_m && n >= c_border && n <= c_last_n)
        {
            // image's end is in the middle of the block and we're outside
            // the image width?
            if (x >= c_width)
            {
                // process data until we get into the image
                int j;
#pragma unroll
                for (j=x; j>=c_width; --j, --bdata)
                {
                    prev.y = *bdata*b0_2 - prev.x*c_a1 - prev.y*c_a2;

                    if (p_fusion)
                        *bdata = prev.y;

                    swap(prev.x, prev.y);
                }

                // now we're inside the image, we must write to transp_out
                float *out = g_transp_out + (c_width-1)*out_stride + y;

                int mmin = x-(WS-1);

#pragma unroll
                for (;j>=mmin; --j, --bdata, out -= out_stride)
                {
                    prev.y = *bdata*b0_2 - prev.x*c_a1 - prev.y*c_a2;

                    if (p_fusion)
                        *bdata = prev.y;

                    if (y < c_height)
                        *out = prev.y;

                    swap(prev.x, prev.y);
                }
            }
            else
            {
                float *out = g_transp_out + x*out_stride + y;

#pragma unroll
                for (int j=WS-1; j>=0; --j, --bdata, out -= out_stride)
                {
                    prev.y = *bdata*b0_2 - prev.x*c_a1 - prev.y*c_a2;

                    if (p_fusion)
                        *bdata = prev.y;

                    if (y < c_height)
                        *out = prev.y;
                    swap(prev.x, prev.y);
                }
            }
        }
        else
        {
#pragma unroll
            for (int j=WS-1; j>=0; --j, --bdata)
            {
                prev.y = *bdata*b0_2 - prev.x*c_a1 - prev.y*c_a2;

                if (p_fusion)
                    *bdata = prev.y;

                swap(prev.x, prev.y);
            }
        }

        if (p_fusion)
        {
            g_pubar += n*c_carry_width + m*WS + tx; 
            g_evhat += n*c_carry_width + m*WS + tx;

            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty*WS][tx];

            // calculate pubar, scan left -> right
            float2 prev = make_float2(0,**bdata++);

#pragma unroll
            for (int i=1; i<WS; ++i, ++bdata)
            {
                **bdata = prev.x = **bdata - prev.y*c_a1 - prev.x*c_a2;

                swap(prev.x, prev.y);
            }

            if (n < c_n_size-1)
                *g_pubar = prev*c_b0;

            if (n > 0)
            {
                // calculate evhat, scan right -> left
                prev = make_float2(**--bdata, 0);

                --bdata;

#pragma unroll
                for (int i=WS-2; i>=0; --i, --bdata)
                {
                    prev.y = **bdata - prev.x*c_a1 - prev.y*c_a2;
                    swap(prev.x, prev.y);
                }

                *g_evhat = prev*b0_2;
            }
        }
    }
}

//-- Algorithm 4_2 Stage 2 and 3 ----------------------------------------------

__global__  __launch_bounds__(WS*DW, DNB)
void alg4_stage2_3( float2 *g_transp_pybar,
                    float2 *g_transp_ezhat ) {

    alg4_stage2_3v5_6( g_transp_pybar, g_transp_ezhat );

}

//-- Algorithm 4_2 Stage 4 ----------------------------------------------------

__global__ __launch_bounds__(WS*DW, DNB)
void alg4_stage4( float *g_transp_out,
                  float2 *g_transp_py,
                  float2 *g_transp_ez,
                  float2 *g_pubar,
                  float2 *g_evhat,
                  int out_stride ) {

    alg4_stage4v7<true>( g_transp_out, g_transp_py, g_transp_ez, g_pubar,
                         g_evhat, out_stride );

}

//-- Algorithm 4_2 Stage 5 and 6 ----------------------------------------------

__global__  __launch_bounds__(WS*DW, DNB)
void alg4_stage5_6( float2 *g_transp_pybar,
                    float2 *g_transp_ezhat ) {

    alg4_stage2_3v5_6( g_transp_pybar, g_transp_ezhat );

}

//-- Algorithm 4_2 Stage 7 ----------------------------------------------------

__global__ __launch_bounds__(WS*DW, DNB)
void alg4_stage7( float *g_out,
                  float2 *g_transp_py,
                  float2 *g_transp_ez,
                  int out_stride ) {

    alg4_stage4v7<false>( g_out, g_transp_py, g_transp_ez, 0, 0,
                          out_stride );

}

//-- Host ---------------------------------------------------------------------

__host__
inline int transp_out_height( const int& h ) {
    // cudaBindTexture2D chokes when memory block stride isn't
    // multiple of 256 bytes, let's add some padding.
    return ((h+WS-1)/WS)*WS;
}

__host__
void prepare_alg4( alg_setup& algs,
                   alg_setup& algs_transp,
                   dvector<float>& d_out,
                   dvector<float>& d_transp_out,
                   dvector<float2>& d_transp_pybar,
                   dvector<float2>& d_transp_ezhat,
                   dvector<float2>& d_pubar,
                   dvector<float2>& d_evhat,
                   cudaArray *& a_in,
                   const float *h_in,
                   const int& w,
                   const int& h,
                   const float& b0,
                   const float& a1,
                   const float& a2,
                   const int& extb,
                   const initcond& ic )
{

    up_constants_coefficients2( b0, a1, a2 );

    calc_alg_setup( algs, w, h, extb );
    calc_alg_setup( algs_transp, h, w, extb );

    d_out.resize( w * h );

    d_transp_out.resize( transp_out_height(h) * w );

    d_transp_pybar.resize( algs.m_size * algs.carry_height );
    d_transp_ezhat.resize( algs.m_size * algs.carry_height );
    d_pubar.resize( algs.n_size * algs.carry_width );
    d_evhat.resize( algs.n_size * algs.carry_width );

    d_transp_pybar.fill_zero();
    d_transp_ezhat.fill_zero();
    d_pubar.fill_zero();
    d_evhat.fill_zero();

    up_texture( a_in, h_in, w, h, ic );

}

__host__
void alg4( dvector<float>& d_out,
           dvector<float>& d_transp_out,
           dvector<float2>& d_transp_pybar,
           dvector<float2>& d_transp_ezhat,
           dvector<float2>& d_pubar,
           dvector<float2>& d_evhat,
           const cudaArray *a_in,
           const alg_setup& algs,
           const alg_setup& algs_transp )
{

    dvector<float2> d_transp_py, d_transp_ez, d_pu, d_ev;

    cudaBindTextureToArray( t_in, a_in );

    up_alg_setup( algs );

    alg4_stage1<<<
        dim3((algs.m_size+2-1)/2, algs.n_size), dim3(WS, DW) >>>(
            d_transp_pybar, d_transp_ezhat );

    alg4_stage2_3<<<
        dim3(1, algs.n_size), dim3(WS, std::min<int>(algs.m_size, DW)) >>>(
            d_transp_pybar, d_transp_ezhat );

    swap( d_transp_pybar, d_transp_py );
    swap( d_transp_ezhat, d_transp_ez );

    alg4_stage4<<<
        dim3((algs.m_size+2-1)/2, algs.n_size), dim3(WS, DW) >>>(
            d_transp_out, d_transp_py, d_transp_ez, d_pubar, d_evhat,
            transp_out_height(algs.height) );

    up_alg_setup( algs_transp );

    alg4_stage5_6<<<
        dim3(1, algs.m_size), dim3(WS, std::min<int>(algs.n_size, DW)) >>>(
            d_pubar, d_evhat );

    swap( d_pubar, d_pu );
    swap( d_evhat, d_ev );

    cudaUnbindTexture( t_in );

    size_t offset;
    cudaBindTexture2D( &offset, t_in, d_transp_out, algs.height, algs.width,
                       transp_out_height(algs.height)*sizeof(float) );

    alg4_stage7<<<
        dim3((algs.n_size+2-1)/2, algs.m_size), dim3(WS, DW) >>>(
            d_out, d_pu, d_ev, algs.width );

    swap( d_ev, d_evhat );
    swap( d_pu, d_pubar );
    swap( d_transp_ez, d_transp_ezhat );
    swap( d_transp_py, d_transp_pybar );

    cudaUnbindTexture( t_in );

}

__host__
void alg4( float *h_inout,
           const int& w,
           const int& h,
           const float& b0,
           const float& a1,
           const float& a2,
           const int& extb,
           const initcond& ic )
{

    alg_setup algs, algs_transp;
    dvector<float> d_out, d_transp_out;
    dvector<float2> d_transp_pybar, d_transp_ezhat, d_pubar, d_evhat;
    cudaArray *a_in;

    prepare_alg4( algs, algs_transp, d_out, d_transp_out, d_transp_pybar,
                  d_transp_ezhat, d_pubar, d_evhat, a_in, h_inout, w, h,
                  b0, a1, a2, extb, ic );

    alg4( d_out, d_transp_out, d_transp_pybar, d_transp_ezhat, d_pubar,
          d_evhat, a_in, algs, algs_transp );

    d_out.copy_to( h_inout, w * h );

    cudaFreeArray( a_in );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
// vi: ai ts=4 sw=4
