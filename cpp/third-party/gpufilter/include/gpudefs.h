/**
 *  GPU definitions (header)
 *  @author Andre Maximo
 *  @author Rodolfo Lima
 *  @date March, 2011
 */

#ifndef GPUDEFS_H
#define GPUDEFS_H

//== INCLUDES =================================================================

#include <cuda_runtime.h>

#include <defs.h>
#include <extension.h>

//== DEFINITIONS ===============================================================

#define HWS 16 // Half Warp Size
#define DW 8 // Default number of warps (computational block height)
#define CHW 7 // Carry-heavy number of warps (computational block height for some kernels)
#define OW 6 // Optimized number of warps (computational block height for some kernels)
#define DNB 6 // Default number of blocks per SM (minimum blocks per SM launch bounds)
#define ONB 5 // Optimized number of blocks per SM (minimum blocks per SM for some kernels)
#define MTS 192 // Maximum number of threads per block with 8 blocks per SM
#define MBO 8 // Maximum number of blocks per SM using optimize or maximum warps
#define CHB 7 // Carry-heavy number of blocks per SM using default number of warps
#define MW 6 // Maximum number of warps per block with 8 blocks per SM (with all warps computing)
#define SOW 5 // Dual-scheduler optimized number of warps per block (with 8 blocks per SM and to use the dual scheduler with 1 computing warp)
#define MBH 3 // Maximum number of blocks per SM using half-warp size

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @struct _alg_setup gpudefs.h
 *  @ingroup api_gpu
 *  @brief Algorithm setup to configure the GPU to run
 */
typedef struct _alg_setup {
    int width, ///< Image width
        height, ///< Image height
        m_size, ///< Number of column-blocks
        n_size, ///< Number of row-blocks
        last_m, ///< Last valid column-block
        last_n, ///< Last valid row-block
        border, ///< Border extension to consider outside image
        carry_height, ///< Auxiliary carry-image height
        carry_width; ///< Auxiliary carry-image width
    float inv_width, ///< Inverse of image width
        inv_height; ///< Inverse of image height
} alg_setup; ///< @see _alg_setup

//== PROTOTYPES ===============================================================

/**
 *  @ingroup api_gpu
 *  @brief Calculate algorithm setup values
 *
 *  Given the dimensions of the 2D work image, calculate the device
 *  constant memory size-related values.  It returns the setup to run
 *  any GPU algorithm.
 *
 *  @param[out] algs Algorithm setup to be uploaded to the GPU
 *  @param[in] w Width of the work image
 *  @param[in] h Height of the work image
 */
extern
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h );

/**
 *  @ingroup api_gpu
 *  @overload
 *  @brief Upload device constants sizes
 *
 *  Given the dimensions of the 2D work image, calculate the device
 *  constant memory size-related values.  The work image is the
 *  original image plus extension blocks to run algorithms
 *  out-of-bounds.  It returns the setup to run any GPU algorithm.
 *
 *  @param[out] algs Algorithm setup to be uploaded to the GPU
 *  @param[in] w Width of the work image
 *  @param[in] h Height of the work image
 *  @param[in] extb Extension (in blocks) to consider outside image
 */
extern
void calc_alg_setup( alg_setup& algs,
                     const int& w,
                     const int& h,
                     const int& extb );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // GPUDEFS_H
//=============================================================================
