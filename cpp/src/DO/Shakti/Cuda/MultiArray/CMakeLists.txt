file(GLOB SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../MultiArray.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/Cuda/MultiArray)
add_library(DO_Shakti_Cuda_MultiArray INTERFACE)
target_sources(DO_Shakti_Cuda_MultiArray
  INTERFACE "${SRC_FILES}")
target_include_directories(DO_Shakti_Cuda_MultiArray
  INTERFACE
  $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(DO_Shakti_Cuda_MultiArray
  INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,AppleClang,Clang>:-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CUDA,GNU,AppleClang,Clang>:-Xcompiler=-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4068>
  $<$<COMPILE_LANG_AND_ID:CUDA,MSVC>:-Xcompiler=/wd4068>)
add_library(DO::Shakti::Cuda::MultiArray ALIAS DO_Shakti_Cuda_MultiArray)
