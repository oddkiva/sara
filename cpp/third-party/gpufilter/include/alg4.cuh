/**
 *  @file alg4.cuh
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 4
 *  @author Rodolfo Lima
 *  @date September, 2011
 */

#ifndef ALG4_CUH
#define ALG4_CUH

//== NAMESPACES ===============================================================

namespace gpufilter {

//== PROTOTYPES ===============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 1
 *
 *  This function computes the algorithm stage 4.1 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, compute and store the
 *  \f$P_{m,n}(\bar{Y})\f$ and \f$E_{m,n}(\hat{Z})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[out] g_transp_pybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[out] g_transp_ezhat All \f$E_{m,n}(\hat{Z})\f$
 */
__global__
void alg4_stage1( float2 *g_transp_pybar,
                  float2 *g_transp_ezhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 2 and 3 (fusioned)
 *
 *  This function computes the algorithm stages 4.2 and 4.3 following:
 *
 *  \li Sequentially for each \f$m\f$, but in parallel for each
 *  \f$n\f$, compute and store the \f$P_{m,n}(Y)\f$ and using the
 *  previously computed \f$P_{m,n}(\bar{Y})\f$.
 *
 *  \li Sequentially for each \f$m\f$, but in parallel for each
 *  \f$n\f$, compute and store the \f$E_{m,n}(Z)\f$ using the
 *  previously computed \f$P_{m-1,n}(Y)\f$ and \f$E_{m,n}(\hat{Z})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[in,out] g_transp_pybar All \f$P_{m,n}(\bar{Y})\f$ fixed to \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_transp_ezhat All \f$E_{m,n}(\hat{Z})\f$ fixed to \f$E_{m,n}(Z)\f$
 */
__global__
void alg4_stage2_3( float2 *g_transp_pybar,
                    float2 *g_transp_ezhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 4
 *
 *  This function computes the algorithm stage 4.4 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, compute
 *  \f$B_{m,n}(Y)\f$ using the previously computed
 *  \f$P_{m-1,n}(Y)\f$. Then compute and store the \f$B_{m,n}(Z)\f$
 *  using the previously computed \f$E_{m+1,n}(Z)\f$.  Finally,
 *  compute and store both \f$P^T_{m,n}(\bar{U})\f$ and
 *  \f$E^T_{m,n}(\hat{V})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[out] g_transp_out The output 2D image transposed
 *  @param[in] g_transp_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_ez All \f$E_{m,n}(Z)\f$
 *  @param[out] g_pubar All \f$P^T_{m,n}(\bar{U})\f$
 *  @param[out] g_evhat All \f$E^T_{m,n}(\hat{V})\f$
 *  @param[in] out_stride Transposed output image stride
 */
__global__
void alg4_stage4( float *g_transp_out,
                  float2 *g_transp_py,
                  float2 *g_transp_ez,
                  float2 *g_pubar,
                  float2 *g_evhat,
                  int out_stride );

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 5 and 6 (fusioned)
 *
 *  This function computes algorithm stages 4.5 and 4.6 following:
 *
 *  \li Sequentially for each \f$n\f$, but in parallel for each
 *  \f$m\f$, compute and store the \f$P^T_{m,n}(U)\f$ from
 *  \f$P^T_{m,n}(\bar{U})\f$.
 *
 *  \li Sequentially for each \f$n\f$, but in parallel for each
 *  \f$m\f$, compute and store each \f$E^T_{m,n}(V)\f$ using the
 *  previously computed \f$P^T_{m,n-1}(U)\f$ and
 *  \f$E^T_{m,n}(\hat{V})\f$.
 *
 *  @note This function is exactly the same as alg4_stage2_3() given
 *  the intrinsic similarities of the kernels.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[in,out] g_transp_pybar All \f$P_{m,n}(\bar{Y})\f$ fixed to \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_transp_ezhat All \f$E_{m,n}(\hat{Z})\f$ fixed to \f$E_{m,n}(Z)\f$
 */
__global__
void alg4_stage5_6( float2 *g_transp_pybar,
                    float2 *g_transp_ezhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm 4 stage 7
 *
 *  This function computes the algorithm stage 4.7 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, compute
 *  \f$B_{m,n}(V)\f$ using the previously computed
 *  \f$P^T_{m,n-1}(V)\f$ and \f$B_{m,n}(Z)\f$.  Then compute and store
 *  the \f$B_{m,n}(U)\f$ using the previously computed
 *  \f$E_{m,n+1}(U)\f$.
 *
 *  @note This function is exactly the same as alg4_stage4() given the
 *  intrinsic similarities of the kernels.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_transp_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] out_stride Output image stride
 */
__global__
void alg4_stage7( float *g_out,
                  float2 *g_transp_py,
                  float2 *g_transp_ez,
                  int out_stride );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // ALG4_CUH
//=============================================================================
