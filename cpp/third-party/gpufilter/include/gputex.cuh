/**
 *  GPU textures
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUTEX_CUH
#define GPUTEX_CUH

//== INCLUDES =================================================================

#include <cuda_runtime.h>

#include <extension.h>

//== DEFINITIONS ===============================================================

// Naming conventions are: c_ constant; t_ texture; g_ global memory;
// s_ shared memory; d_ device pointer; a_ cuda-array; p_ template
// parameter in kernels; f_ surface; h_ host pointer.

texture< float, cudaTextureType2D, cudaReadModeElementType > t_in;

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

//-- Host ---------------------------------------------------------------------

/**
 *  @ingroup api_gpu
 *  @brief Upload input image as a texture in device
 *
 *  Given an input image in the host, upload it to the device memory
 *  as a texture.
 *
 *  @param[out] a_in The input 2D image as cudaArray to be allocated and copied to device memory
 *  @param[in] h_in The input 2D image to compute algorithm 5 in host memory
 *  @param[in] w Image width
 *  @param[in] h Image height
 *  @param[in] ic Initial condition (for outside access) (default zero)
 */
__host__
void up_texture( cudaArray *& a_in,
                 const float *h_in,
                 const int& w,
                 const int& h,
                 const initcond& ic ) {

    // cuda channel descriptor for texture
    cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
    cudaMallocArray( &a_in, &ccd, w, h );
    cudaMemcpyToArray( a_in, 0, 0, h_in, w*h*sizeof(float),
                       cudaMemcpyHostToDevice );

    t_in.normalized = true;
    t_in.filterMode = cudaFilterModePoint;

    switch( ic ) {
    case zero: // mode border defaults to zero-border
        t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeBorder;
        break;
    case clamp:
        t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeClamp;
        break;
    case repeat: // mode wrap implements repeat
        t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeWrap;
        break;
    case mirror:
        t_in.addressMode[0] = t_in.addressMode[1] = cudaAddressModeMirror;
        break;
    }

}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUTEX_CUH
//=============================================================================
