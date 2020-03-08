project(gpufilter CXX C)

if (NOT USE_SM)
    # set(USE_SM 30) #using SM_35 in TITAN was slower than SM_30
    # set(USE_SM 35) #..it does not seem to be the case anymore..
    set(USE_SM 61) # Pascal
endif ()

set(CUDA_NVCC_FLAGS
  --ptxas-options=-v -O3
  -ftz=true -prec-div=false -prec-sqrt=false
  -use_fast_math #-maxrregcount=255
  -arch=sm_${USE_SM}
)

set (INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_subdirectory(src)
