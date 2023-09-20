file(GLOB DO_Sara_OpenCL_FILES FILES *.hpp)
set_source_files_properties(${DO_Sara_OpenCL_FILES} PROPERTIES LANGUAGE CXX)

add_library(DO_Sara_OpenCL INTERFACE)
add_library(DO::Sara::OpenCL ALIAS DO_Sara_OpenCL)
target_sources(DO_Sara_OpenCL INTERFACE ${DO_Sara_OpenCL_FILES})

target_include_directories(
  DO_Sara_OpenCL INTERFACE $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
  DO_Sara_OpenCL #
  INTERFACE DO::Sara::Core #
            OpenCL::OpenCL)

target_compile_definitions(
  DO_Sara_OpenCL
  INTERFACE $<$<PLATFORM_ID:Darwin>:CL_SILENCE_DEPRECATION> #
            CL_USE_DEPRECATED_OPENCL_1_0_APIS #
            CL_USE_DEPRECATED_OPENCL_1_1_APIS #
            CL_USE_DEPRECATED_OPENCL_1_2_APIS)
if(CMAKE_CUDA_COMPILER)
  # Disable compiler warnings with CUDA.
  # https://developer.nvidia.com/blog/nvidia-is-now-opencl-3-0-conformant/
  target_compile_definitions(DO_Sara_OpenCL
                             INTERFACE CL_TARGET_OPENCL_VERSION=300)
endif()

target_compile_features(DO_Sara_OpenCL INTERFACE cxx_std_17)

add_subdirectory(examples)
add_subdirectory(test)
