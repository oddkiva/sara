/**
 *  @file dvector.h
 *  @brief Device Vector utility class definition
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date February, 2011
 */

#ifndef DVECTOR_H
#define DVECTOR_H

//== INCLUDES =================================================================

#include <vector>

#include <alloc.h>

//== NAMESPACES ===============================================================

namespace gpufilter {

//== CLASS DEFINITION =========================================================

/**
 *  @class dvector dvector.h
 *  @ingroup utils
 *  @brief Device Vector class
 *
 *  Device vector is a STL-based vector in the GPU memory.
 *
 *  @tparam T Device vector values type
 */
template< class T >
class dvector {

public:

    /**
     *  Constructor
     *  @param[in] that Host (STL) Vector data (non-converted) to be copied into this object
     */
    explicit dvector( const std::vector<T>& that ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        *this = that;
    }

    /**
     *  Constructor
     *  @param[in] data Vector data to be copied into this object
     *  @param[in] size Vector data size
     */
    dvector( const T *data,
             const size_t& _size ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        copy_from( data, _size );
    }

    /**
     *  Constructor
     *  @param[in] data Vector data in 2D to be copied into this object
     *  @param[in] w_data Width of the vector data
     *  @param[in] h_data Height of the vector data
     *  @param[in] w Width of the vector data in device memory
     *  @param[in] h Height of the vector data in device memory
     */
    dvector( const T *data,
             const size_t& w_data,
             const size_t& h_data,
             const size_t& w,
             const size_t& h ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        copy_from( data, w_data, h_data, w, h );
    }

    /**
     *  Copy Constructor
     *  @param[in] that Copy that object to this object
     */
    dvector( const dvector<T>& that ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        *this = that;
    }

    /**
     *  Default Constructor
     *  @param[in] size Vector data size
     */
    dvector( const size_t& _size = 0 ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        resize(_size);
    }

    /**
     *  Default Constructor
     *  @param[w] width Width of the 2D array vector
     *  @param[h] height Width of the 2D array vector
     */
    dvector( const size_t& w, const size_t& h ) : m_size(0), m_capacity(0), m_data(0), m_pitch(0) {
        resize(w, h);
    }

    /**
     *  Destructor
     */
    ~dvector() {
        cuda_delete(m_data);
        m_data = 0;
        m_capacity = 0;
        m_size = 0;
    }

    /**
     *  @brief Resize this vector
     *  @param[in] size The new vector size
     */
    void resize( const size_t& _size ) {
        if( _size > m_capacity )
        {
            cuda_delete(m_data);
            m_data = 0;
            m_capacity = 0;
            m_size = 0;

            m_data = cuda_new<T>(_size);
            m_capacity = _size;
            m_size = _size;
        }
        else
            m_size = _size;
    }

    /**
     *  @brief Resize this vector
     *  @param[w] w The width of the new 2D array vector 
     *  @param[h] h The height of the new 2D array vector 
     */
    void resize( const size_t& w, const size_t& h ) {
        cuda_delete(m_data);
        m_data = 0;
        m_capacity = 0;
        m_size = 0;

        m_data = cuda_new<T>(m_pitch, w, h);
        m_capacity = m_pitch*h;
        m_size = m_pitch*h;
    }

    /**
     *  @brief Clear this vector
     */
    void clear() {
        m_size = 0;
    }

    /**
     *  @brief Read/write operator
     *  @param[in] idx Index of vector value
     *  @return Vector value at index
     */
    T operator [] ( const int& idx ) const {
        T value;
        cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    /**
     *  @brief Assign operator
     *  @param[in] that Device vector to copy from
     *  @return This device vector with assigned values
     */
    dvector& operator = ( const dvector<T>& that ) {
        resize(that.size());
        cudaMemcpy(data(), that.data(), size()*sizeof(T), cudaMemcpyDeviceToDevice);
        check_cuda_error("Error during memcpy from device to device");
        return *this;
    }

    /**
     *  @brief Assign operator
     *  @param[in] that Host (STL) Vector to copy from
     *  @return This device vector with assigned values
     */
    dvector& operator = ( const std::vector<T>& that ) {
        resize(that.size());
        cudaMemcpy(data(), &that[0], size()*sizeof(T), cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy from host to device");
        return *this;
    }

    /**
     *  @brief Copy values from this vector to a host (CPU) vector
     *  @param[out] data Host Vector to copy values to
     *  @param[in] s Maximum number of elements to copy
     */
    void copy_to( T *_data,
                  const size_t& s ) const {
        cudaMemcpy(_data, this->data(),
                   std::min(size(),s)*sizeof(T),
                   cudaMemcpyDeviceToHost);
        check_cuda_error("Error during memcpy from device to host");
    }

    /**
     *  @brief Copy values from this vector in 2D to a host (CPU) vector
     *  @param[out] data Host Vector to copy values to
     *  @param[in] w Width of the vector data in device memory
     *  @param[in] h Height of the vector data in device memory
     *  @param[in] w_data Width of the vector data to copy values to
     *  @param[in] h_data Height of the vector data to copy values to
     */
    void copy_to( T *_data,
                  const size_t& w,
                  const size_t& h,
                  const size_t& w_data,
                  const size_t& h_data ) const {
        cudaMemcpy2D(_data, w_data*sizeof(T),
                     this->data(), m_pitch,
                     w_data*sizeof(T), h_data,
                     cudaMemcpyDeviceToHost);
        check_cuda_error("Error during memcpy2D from device to host");
    }

    /**
     *  @brief Copy values from a host (CPU) vector to this vector
     *  @param[in] data Vector data to be copied into this object
     *  @param[in] size Vector data size
     */
    void copy_from( const T *_data,
                    const size_t& _size ) {
        resize(_size);
        cudaMemcpy(this->data(), _data,
                   _size*sizeof(T), cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy from host to device");
    }

    /**
     *  @brief Copy values from a 2D host (CPU) vector to this vector
     *  @param[in] data Vector data in 2D to be copied into this object
     *  @param[in] w_data Width of the vector data
     *  @param[in] h_data Height of the vector data
     *  @param[in] w Width of the vector data in device memory
     *  @param[in] h Height of the vector data in device memory
     */
    void copy_from( const T *_data,
                    const size_t& w_data,
                    const size_t& h_data,
                    const size_t& w,
                    const size_t& h ) {
        resize(w, h);
        cudaMemcpy2D(this->data(), m_pitch,
                     _data, w_data*sizeof(T),
                     w_data*sizeof(T), h_data,
                     cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy2D from host to device");
    }

    /**
     *  @brief Fill this vector with zeroes
     */
    void fill_zero() {
       cudaMemset(m_data, 0, m_size*sizeof(T));
    }

    /**
     *  @brief Check if this vector is empty
     *  @return True if this vector is empty
     */
    bool empty() const { return size()==0; }

    /**
     *  @brief Size of this vector
     *  @return Vector size
     */
    size_t size() const { return m_size; }

    /**
     *  @brief Data in this vector
     *  @return Vector data
     */
    T *data() { return m_data; }

    /**
     *  @overload
     *  @return Constant vector data
     */
    const T *data() const { return m_data; }

    /**
     *  @brief Get last element of the vector
     *  @return Last element of this vector
     */
    T back() const { return operator[](size()-1); }

    /**
     *  @brief Address access operator
     *  @return Pointer to vector data
     */
    operator T* () { return data(); }

    /**
     *  @brief Address access operator
     *  @return Constant pointer to vector data
     */
    operator const T* () const { return data(); }

    /**
     *  @brief Swap vector values
     *  @param[in,out] a Vector to be swapped
     *  @param[in,out] b Vector to be swapped
     */
    friend void swap( dvector<T>& a,
                      dvector<T>& b ) {
        std::swap(a.m_data, b.m_data);
        std::swap(a.m_size, b.m_size);
        std::swap(a.m_capacity, b.m_capacity);
    }

private:

    T *m_data; ///< Vector data
    size_t m_size; ///< Vector size
    size_t m_capacity; ///< Vector capacity
    size_t m_pitch; ///< Pitch for 2D arrays

};

//== IMPLEMENTATION ===========================================================

/**
 *  @relates dvector
 *  @brief Copy to the CPU a vector in the GPU
 *
 *  This function copies a device vector (GPU) to a host vector (CPU).
 *
 *  @param[in] d_vec Pointer to the device vector (in the GPU memory)
 *  @param[in] len Length of the device vector
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template< class T >
std::vector<T> to_cpu( const T *d_vec,
                       unsigned len ) {
    std::vector<T> out;
    out.resize(len);

    cudaMemcpy(&out[0], d_vec, len*sizeof(T), cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");

    return out;
}

/**
 *  @relates dvector
 *  @overload
 *
 *  @param[in] v Device vector (in the GPU memory)
 *  @return Host vector (in the CPU memory) as a STL vector
 *  @tparam T Vector values type
 */
template< class T >
std::vector<T> to_cpu( const dvector<T>& v ) {
    return to_cpu(v.data(), v.size());
}

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // DVECTOR_H
//=============================================================================
//vi: ai sw=4 ts=4
