/**
 *  @file alg5.cuh
 *  @brief CUDA device code for GPU-Efficient Recursive Filtering Algorithm 5
 *  @author Rodolfo Lima
 *  @date September, 2011
 */

#ifndef ALG5_CUH
#define ALG5_CUH

//== NAMESPACES ===============================================================

namespace gpufilter {

//== PROTOTYPES ===============================================================

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 1
 *
 *  This function computes the algorithm stage 5.1 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, compute and store
 *  each \f$P_{m,n}(\bar{Y})\f$, \f$E_{m,n}(\hat{Z})\f$,
 *  \f$P^T_{m,n}(\check{U})\f$, and \f$E^T_{m,n}(\tilde{V})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[out] g_transp_pybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[out] g_transp_ezhat All \f$E_{m,n}(\hat{Z})\f$
 *  @param[out] g_ptucheck All \f$P^T_{m,n}(\check{U})\f$
 *  @param[out] g_etvtilde All \f$E^T_{m,n}(\tilde{V})\f$
 */
__global__
void alg5_stage1( float *g_transp_pybar,
                  float *g_transp_ezhat,
                  float *g_ptucheck,
                  float *g_etvtilde );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 2 and 3 (fusioned)
 *
 *  This function computes the algorithm stages 5.2 and 5.3 following:
 *
 *  \li In parallel for all \f$n\f$, sequentially for each \f$m\f$,
 *  compute and store the \f$P_{m,n}(Y)\f$ and using the previously
 *  computed \f$P_{m,n}(\bar{Y})\f$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  \li In parallel for all \f$n\f$, sequentially for each \f$m\f$,
 *  compute and store \f$E_{m,n}(Z)\f$ using the previously computed
 *  \f$P_{m-1,n}(Y)\f$ and \f$E_{m+1,n}(\hat{Z})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[in,out] g_transp_pybar All \f$P_{m,n}(\bar{Y})\f$ fixed to \f$P_{m,n}(Y)\f$
 *  @param[in,out] g_transp_ezhat All \f$E_{m,n}(\hat{Z})\f$ fixed to \f$E_{m,n}(Z)\f$
 */
__global__
void alg5_stage2_3( float *g_transp_pybar,
                    float *g_transp_ezhat );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 4 and 5 (fusioned)
 *
 *  This function computes the algorithm stages 5.4 and 5.5 following:
 *
 *  \li In parallel for all \f$m\f$, sequentially for each \f$n\f$,
 *  compute and store \f$P^T_{m,n}(U)\f$ and using the previously
 *  computed \f$P^T_{m,n}(\check{U})\f$, \f$P_{m-1,n}(Y)\f$, and
 *  \f$E_{m+1,n}(Z)\f$.
 *
 *  with simple kernel fusioned (going thorough global memory):
 *
 *  \li In parallel for all \f$m\f$, sequentially for each \f$n\f$,
 *  compute and store \f$E^T_{m,n}(V)\f$ and using the previously
 *  computed \f$E^T_{m,n}(\tilde{V})\f$, \f$P^T_{m,n-1}(U)\f$,
 *  \f$P_{m-1,n}(Y)\f$, and \f$E_{m+1,n}(Z)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[in,out] g_ptucheck All \f$P^T_{m,n}(\check{U})\f$ fixed to \f$P^T_{m,n}(\bar{U})\f$
 *  @param[in,out] g_etvtilde All \f$E^T_{m,n}(\tilde{V})\f$ fixed to \f$E^T_{m,n}(\check{V})\f$
 *  @param[in] g_transp_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_ez All \f$E_{m,n}(Z)\f$
 */
__global__
void alg5_stage4_5( float *g_ptucheck,
                    float *g_etvtilde,
                    const float *g_transp_py,
                    const float *g_transp_ez );

/**
 *  @ingroup gpu
 *  @brief Algorithm 5 stage 6
 *
 *  This function computes the algorithm stage 5.6 following:
 *
 *  \li In parallel for all \f$m\f$ and \f$n\f$, compute one after the
 *  other \f$B_{m,n}(Y)\f$, \f$B_{m,n}(Z)\f$, \f$B_{m,n}(U)\f$, and
 *  \f$B_{m,n}(V)\f$ and using the previously computed
 *  \f$P_{m-1,n}(Y)\f$, \f$E_{m+1,n}(Z)\f$, \f$P^T_{m,n-1}(U)\f$, and
 *  \f$E^T_{m,n+1}(V)\f$. Store \f$B_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5()
 *  @param[out] g_out The output 2D image
 *  @param[in] g_transp_py All \f$P_{m,n}(Y)\f$
 *  @param[in] g_transp_ez All \f$E_{m,n}(Z)\f$
 *  @param[in] g_ptu All \f$P^T_{m,n}(U)\f$
 *  @param[in] g_etv All \f$E^T_{m,n}(V)\f$
 */
__global__
void alg5_stage6( float *g_out,
                  const float *g_transp_py,
                  const float *g_transp_ez,
                  const float *g_ptu,
                  const float *g_etv );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // ALG5_CUH
//=============================================================================
