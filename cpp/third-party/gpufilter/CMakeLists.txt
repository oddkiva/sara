project(gpufilter CXX C)

if (NOT USE_SM)
  # set(USE_SM 30) # using SM_35 in TITAN was slower than SM_30
  # set(USE_SM 35) # ..it does not seem to be the case anymore..
  set(USE_SM 61) # Pascal
endif ()

add_subdirectory(src)
