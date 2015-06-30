#include_directories(${CMAKE_SOURCE_DIR}/include algorithms util nn .)

add_definitions(-D_FLANN_VERSION=${FLANN_VERSION})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/flann/config.h)


################################################################################
# List the C, C++ and CUDA source files with file globbing.
# 
file(GLOB_RECURSE C_SOURCES flann.cpp)
file(GLOB_RECURSE CPP_SOURCES flann_cpp.cpp)
file(GLOB_RECURSE CU_SOURCES *.cu)


################################################################################
# Create the static C++ library.
# 
add_library(flann_cpp_s STATIC ${CPP_SOURCES})
set_target_properties(flann_cpp_s
  PROPERTIES
  OUTPUT_NAME_DEBUG flann_cpp_s-d)
if(CMAKE_COMPILER_IS_GNUCC)
    set_target_properties(flann_cpp_s PROPERTIES COMPILE_FLAGS -fPIC)
endif()
set_property(TARGET flann_cpp_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)
install(TARGETS flann_cpp_s
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT ThirdParty)


################################################################################
# Create the CUDA-based implementation of FLANN.
# 
if (BUILD_CUDA_LIB)
    SET(CUDA_NVCC_FLAGS -DFLANN_USE_CUDA)
    if(CMAKE_COMPILER_IS_GNUCC)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;-arch=sm_13" )
        if (NVCC_COMPILER_BINDIR)
            set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--compiler-bindir=${NVCC_COMPILER_BINDIR}")
        endif()
    else()
	    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_13" )
    endif()
    cuda_add_library(flann_cuda_s STATIC ${CU_SOURCES})
    set_property(TARGET flann_cuda_s PROPERTY COMPILE_DEFINITIONS FLANN_STATIC)
endif()
