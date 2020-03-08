/**
 *  @file sat.cu
 *  @brief CUDA device code for GPU-Efficient Summed-Area Tables
 *  @author Andre Maximo
 *  @date September, 2011
 */

//== INCLUDES =================================================================

#include <dvector.h>

#include <gpufilter.h>
#include <gpuconsts.cuh>

#include <sat.cuh>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Algorithm SAT Stage 1 ----------------------------------------------------

__global__ __launch_bounds__( WS * SOW, MBO )
void algSAT_stage1( const float *g_in,
                    float *g_ybar,
                    float *g_vhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_in += (row0+ty)*c_width+col;
	g_ybar += by*c_width+col;
	g_vhat += bx*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_in;
    }

	__syncthreads();

	if( ty == 0 ) {

        {   // calculate ybar -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev = **bdata;
            ++bdata;

#pragma unroll
            for (int i = 1; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;

            *g_ybar = prev;
        }

        {   // calculate vhat -----------------------
            float *bdata = s_block[tx];

            float prev = *bdata;
            ++bdata;

#pragma unroll
            for (int i = 1; i < WS; ++i, ++bdata)
                prev = *bdata + prev;

            *g_vhat = prev;
        }

	}

}

//-- Algorithm SAT Stage 2 ----------------------------------------------------

__global__ __launch_bounds__( WS * MW, MBO )
void algSAT_stage2( float *g_ybar,
                    float *g_ysum ) {

	const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, col0 = bx*MW+ty, col = col0*WS+tx;

	if( col >= c_width ) return;

	g_ybar += col;
	float y = *g_ybar;
	int ln = HWS+tx;

	if( tx == WS-1 )
		g_ysum += col0;

	volatile __shared__ float s_block[ MW ][ HWS+WS+1 ];

	if( tx < HWS ) s_block[ty][tx] = 0.f;
	else s_block[ty][ln] = 0.f;

	for (int n = 1; n < c_n_size; ++n) {

        // calculate ysum -----------------------

		s_block[ty][ln] = y;

		s_block[ty][ln] += s_block[ty][ln-1];
		s_block[ty][ln] += s_block[ty][ln-2];
		s_block[ty][ln] += s_block[ty][ln-4];
		s_block[ty][ln] += s_block[ty][ln-8];
		s_block[ty][ln] += s_block[ty][ln-16];

		if( tx == WS-1 ) {
			*g_ysum = s_block[ty][ln];
			g_ysum += c_m_size;
		}

        // fix ybar -> y -------------------------

		g_ybar += c_width;
		y = *g_ybar += y;

	}

}

//-- Algorithm SAT Stage 3 ----------------------------------------------------

__global__ __launch_bounds__( WS * MW, MBO )
void algSAT_stage3( const float *g_ysum,
                    float *g_vhat ) {

	const int tx = threadIdx.x, ty = threadIdx.y,
        by = blockIdx.y, row0 = by*MW+ty, row = row0*WS+tx;

	if( row >= c_height ) return;

	g_vhat += row;
	float y = 0.f, v = 0.f;

	if( row0 > 0 )
		g_ysum += (row0-1)*c_m_size;

	for (int m = 0; m < c_m_size; ++m) {

        // fix vhat -> v -------------------------

		if( row0 > 0 ) {
			y = *g_ysum;
			g_ysum += 1;
		}

		v = *g_vhat += v + y;
		g_vhat += c_height;

	}

}

//-- Algorithm SAT Stage 4 ----------------------------------------------------

__global__ __launch_bounds__( WS * SOW, MBO )
void algSAT_stage4( float *g_inout,
                    const float *g_y,
                    const float *g_v ) {

	const int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_inout += (row0+ty)*c_width+col;
	if( by > 0 ) g_y += (by-1)*c_width+col;
	if( bx > 0 ) g_v += (bx-1)*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_inout;
        bdata += SOW;
        g_inout += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_inout;
    }

	__syncthreads();

	if( ty == 0 ) {

        {   // calculate y -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev;
            if( by > 0 ) prev = *g_y;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;
        }

        {   // calculate x -----------------------
            float *bdata = s_block[tx];

            float prev;
            if( bx > 0 ) prev = *g_v;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                *bdata = prev = *bdata + prev;
        }

	}

	__syncthreads();

    bdata = (float (*)[WS+1]) &s_block[ty][tx];

	g_inout -= (WS-(WS%SOW))*c_width;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        *g_inout = **bdata;
        bdata += SOW;
        g_inout += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        *g_inout = **bdata;
    }

}

//-- Algorithm SAT Stage 4 (not-in-place) -------------------------------------

__global__ __launch_bounds__( WS * SOW, MBO )
void algSAT_stage4( float *g_out,
                    const float *g_in,
                    const float *g_y,
                    const float *g_v ) {

	const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y, col = bx*WS+tx, row0 = by*WS;

	__shared__ float s_block[ WS ][ WS+1 ];

    float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[ty][tx];

	g_in += (row0+ty)*c_width+col;
	if( by > 0 ) g_y += (by-1)*c_width+col;
	if( bx > 0 ) g_v += (bx-1)*c_height+row0+tx;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        **bdata = *g_in;
        bdata += SOW;
        g_in += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        **bdata = *g_in;
    }

	__syncthreads();

	if( ty == 0 ) {

        {   // calculate y -----------------------
            float (*bdata)[WS+1] = (float (*)[WS+1]) &s_block[0][tx];

            float prev;
            if( by > 0 ) prev = *g_y;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                **bdata = prev = **bdata + prev;
        }

        {   // calculate x -----------------------
            float *bdata = s_block[tx];

            float prev;
            if( bx > 0 ) prev = *g_v;
            else prev = 0.f;

#pragma unroll
            for (int i = 0; i < WS; ++i, ++bdata)
                *bdata = prev = *bdata + prev;
        }

	}

	__syncthreads();

    bdata = (float (*)[WS+1]) &s_block[ty][tx];

	g_out += (row0+ty)*c_width+col;

#pragma unroll
    for (int i = 0; i < WS-(WS%SOW); i+=SOW) {
        *g_out = **bdata;
        bdata += SOW;
        g_out += SOW * c_width;
    }
    if( ty < WS%SOW ) {
        *g_out = **bdata;
    }

}

//-- Host ---------------------------------------------------------------------

__host__
void prepare_algSAT( alg_setup& algs,
                     dvector<float>& d_inout,
                     dvector<float>& d_ybar,
                     dvector<float>& d_vhat,
                     dvector<float>& d_ysum,
                     const float *h_in,
                     const int& w,
                     const int& h ) {

    algs.width = w;
    algs.height = h;

    if( w % 32 > 0 ) algs.width += (32 - (w % 32));
    if( h % 32 > 0 ) algs.height += (32 - (h % 32));

    calc_alg_setup( algs, algs.width, algs.height );
    up_alg_setup( algs );

    d_inout.copy_from( h_in, w, h, algs.width, algs.height );

    d_ybar.resize( algs.n_size * algs.width );
    d_vhat.resize( algs.m_size * algs.height );
    d_ysum.resize( algs.m_size * algs.n_size );

}

__host__
void algSAT( dvector<float>& d_out,
             dvector<float>& d_ybar,
             dvector<float>& d_vhat,
             dvector<float>& d_ysum,
             const dvector<float>& d_in,
             const alg_setup& algs ) {

	const int nWm = (algs.width+MTS-1)/MTS, nHm = (algs.height+MTS-1)/MTS;
    const dim3 cg_img( algs.m_size, algs.n_size );
    const dim3 cg_ybar( nWm, 1 );
    const dim3 cg_vhat( 1, nHm );

    algSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_in, d_ybar, d_vhat );

    algSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

    algSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

    algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_out, d_in, d_ybar, d_vhat );

}

__host__
void algSAT( dvector<float>& d_inout,
             dvector<float>& d_ybar,
             dvector<float>& d_vhat,
             dvector<float>& d_ysum,
             const alg_setup& algs ) {

	const int nWm = (algs.width+MTS-1)/MTS, nHm = (algs.height+MTS-1)/MTS;
    const dim3 cg_img( algs.m_size, algs.n_size );
    const dim3 cg_ybar( nWm, 1 );
    const dim3 cg_vhat( 1, nHm );

    algSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_inout, d_ybar, d_vhat );

    algSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

    algSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

    algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_inout, d_ybar, d_vhat );

}

__host__
void algSAT( float *h_inout,
             const int& w,
             const int& h ) {

    alg_setup algs;
    dvector<float> d_out, d_ybar, d_vhat, d_ysum;

    prepare_algSAT( algs, d_out, d_ybar, d_vhat, d_ysum, h_inout, w, h );

    algSAT( d_out, d_ybar, d_vhat, d_ysum, algs );

    d_out.copy_to( h_inout, algs.width, algs.height, w, h );

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
