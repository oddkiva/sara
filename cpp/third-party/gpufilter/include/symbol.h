/**
 *  @file symbol.h
 *  @brief Device Symbol Management definition
 *  @author Rodolfo Lima
 *  @date February, 2011
 */

#ifndef SYMBOL_H
#define SYMBOL_H

//== INCLUDES =================================================================

#include <string>
#include <vector>
#include <cuda.h>

#include <error.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== IMPLEMENTATION ===========================================================

#if CUDA_VERSION >= 5000

/**
 *  @ingroup utils
 *  @brief Copy value(s) to symbol
 *
 *  @param[in] symbol Symbol in device (GPU)
 *  @param[in] value Value to copy to symbol
 *  @tparam T Value type used in the GPU
 */
template <class T>
void copy_to_symbol( const T &symbol,
                     const T &value ) {
    cudaMemcpyToSymbol(symbol, &value, sizeof(T));
    check_cuda_error("Error copying symbol to device");
}

#else // CUDA <5000

/**
 *  @ingroup utils
 *  @brief Copy value(s) to symbol
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] value Value to copy to symbol
 *  @tparam T Value type used in the GPU
 */
template <class T>
void _copy_to_symbol( const std::string &name,
                      const T &value ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    if(sizeof(T) > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(), &value, sizeof(T), 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error1 copying '"+name+"' buffer to device");
}

#define copy_to_symbol(S,V) _copy_to_symbol(#S,V)

#endif // CUDA 5000

#if CUDA_VERSION >= 5000
/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] symbol Symbols in device (GPU)
 *  @param[in] symbol_size Size of the items to be copied
 *  @param[in] items Host (STL) Vector values
 *  @tparam T Vector values type used in the CPU and the GPU
 *  @tparam S Type of the container of vector values
 */
template <class T, class S>
void copy_to_symbol( const T *symbol,
                     const S &symbol_size,
                     const std::vector<T> &items ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, symbol);
    check_cuda_error("Invalid symbol");

    size_t size = items.size()*sizeof(T);

    if(size > size_storage)
        throw std::runtime_error("symbol storage overflow");

    cudaMemcpyToSymbol(symbol,&items[0], size, 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying symbol to device");

    copy_to_symbol(symbol_size, items.size());
}

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] symbol Symbols in device (GPU)
 *  @param[in] beg Iterator to the beginning of symbols
 *  @param[in] end Iterator to the end of symbols
 *  @tparam T Vector values type used in the CPU and the GPU
 *  @tparam IT Iterator type
 */
template <class T, class IT>
void copy_to_symbol( const T *symbol,
                     IT beg,
                     IT end ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, symbol);
    check_cuda_error("Invalid symbol");

    using std::distance;
    size_t size = distance(beg,end)*sizeof(T);

    if(size > size_storage)
        throw std::runtime_error("symbol storage overflow");

    cudaMemcpyToSymbol(symbol,&*beg, size, 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying symbol to device");
}

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] symbol Symbols in device (GPU)
 *  @param[in] items Container with symbols items
 *  @tparam C Container class type
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template <template <class> class C, class T>
void copy_to_symbol( const T *symbol,
                     const C<T> &items ) {
    copy_to_symbol(symbol, items.begin(), items.end());
}

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] symbol Symbols in device (GPU)
 *  @param[in] items Container with symbols items
 *  @tparam C Container class type
 *  @tparam T Vector values type used in the CPU and the GPU
 *  @tparam U Second template class for container
 */
template <template <class,class> class C, class T, class U>
void copy_to_symbol( const T *symbol,
                     const C<T,U> &items ) {
    copy_to_symbol(symbol, items.begin(), items.end());
}

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] symbol Symbols in device (GPU)
 *  @param[in] items Container with symbols items
 *  @tparam C Container class type
 *  @tparam T Vector values type used in the CPU and the GPU
 *  @tparam U Second template class for container is integer
 */
template <template <class,int> class C, class T, int U>
void copy_to_symbol( const T *symbol,
                     const C<T,U> &items ) {
    copy_to_symbol(symbol, items.begin(), items.end());
}

#else // CUDA 5000

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] size_name Name of the symbol storing the size of the items to be copied
 *  @param[in] items Host (STL) Vector values
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template <class T>
void _copy_to_symbol( const std::string &name,
                      const std::string &size_name,
                      const std::vector<T> &items ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    size_t size = items.size()*sizeof(T);

    if(size > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    cudaMemcpyToSymbol(name.c_str(),&items[0], size, 0,
                       cudaMemcpyHostToDevice);
    check_cuda_error("Error copying '"+name+"' buffer to device");

    if(!size_name.empty())
        _copy_to_symbol(size_name.c_str(), items.size());
}

/**
 *  @ingroup utils
 *  @overload
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @param[in] items Host (STL) Vector values
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template <class T>
void _copy_to_symbol( const std::string &name,
                      const std::vector<T> &items ) {
    _copy_to_symbol(name, "", items);
}

#define copy_array_to_symbol(S,SN,V) _copy_to_symbol(#S,#SN,V)

#endif // CUDA 5000

#if CUDA_VERSION >= 5000

/**
 *  @ingroup utils
 *  @brief Copy value(s) from symbol
 *
 *  @param[in] symbol Symbol in device (GPU)
 *  @return Value of the symbol
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template <class T>
T copy_from_symbol( const T &symbol ) {
    T value;
    cudaMemcpyFromSymbol(&value, symbol, sizeof(T), 0,
                         cudaMemcpyDeviceToHost);
    check_cuda_error("Error copying symbol from device");

    return value;
}

#else // CUDA 5000

/**
 *  @ingroup utils
 *  @brief Copy value(s) from symbol
 *
 *  @param[in] name Name of the symbol in device (GPU)
 *  @return Value of the symbol
 *  @tparam T Vector values type used in the CPU and the GPU
 */
template <class T>
T _copy_from_symbol( const std::string &name ) {
    size_t size_storage;
    cudaGetSymbolSize(&size_storage, name.c_str());
    check_cuda_error("Invalid symbol '"+name+"'");

    if(sizeof(T) > size_storage)
        throw std::runtime_error("'"+name+"'"+" storage overflow");

    T value;
    cudaMemcpyFromSymbol(&value, name.c_str(), sizeof(T), 0,
                         cudaMemcpyDeviceToHost);
    check_cuda_error("Error copying '"+name+"' buffer to device");

    return value;
}

#define copy_from_symbol(N) _copy_from_symbol(#N)

#endif // CUDA 5000

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // SYMBOL_H
//=============================================================================
//vi: ai sw=4 ts=4
