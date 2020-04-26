/**
 *  @file error.h
 *  @brief Device Error Management definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef ERROR_H
#define ERROR_H

//== INCLUDES =================================================================

#include <sstream>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

/**
 *  @ingroup utils
 *  @brief Check error in device
 *
 *  This function checks if there is a device error.
 *
 *  @param[in] msg Message to appear in case of a device error
 */
inline void check_cuda_error( const std::string &msg ) {
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess ) {
        if( msg.empty() )
            throw std::runtime_error(cudaGetErrorString(err));
        else {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str().c_str());
        }
    }
}

/**
 *  @brief Check device computation
 *
 *  This function checks if the values computed by the device (GPU)
 *  differ from the values computed by the CPU.  The device values are
 *  called result (res) and the CPU values are called reference (ref).
 *  This function is for debug only.
 *
 *  @param[in] ref Reference (CPU) values
 *  @param[in] res Result (GPU) values
 *  @param[in] ne Number of elements to compare
 *  @param[out] me Maximum error (difference among all values)
 *  @param[out] mre Maximum relative error (difference among all values)
 *  @tparam T1 Values type used in the GPU
 *  @tparam T2 Values type used in the CPU
 */
template< class T1, class T2 >
void check_cpu_reference(const T1 *ref, const T2 *res, const int& ne, T1& me, T1& mre)
{
    mre = me = (T1)0;
    for (int i = 0; i < ne; i++)
    {
        T1 a = (T1)(res[i]) - ref[i];
        if( a < (T1)0 ) a = -a;
        if( ref[i] != (T1)0 )
        {
            T1 r = (ref[i] < (T1)0) ? -ref[i] : ref[i];
            T1 b = a / r;
            mre = b > mre ? b : mre;
        }
        me = a > me ? a : me;
    }
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // ERROR_H
//=============================================================================
//vi: ai sw=4 ts=4
