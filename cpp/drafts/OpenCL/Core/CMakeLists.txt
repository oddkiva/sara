file(GLOB DO_Sara_OpenCL_Core_FILES FILES *.hpp)
set_source_files_properties(${DO_Sara_OpenCL_Core}
  PROPERTIES
  LANGUAGE CXX)

add_library(DO_Sara_OpenCL_Core INTERFACE)
target_sources(DO_Sara_OpenCL_Core
  INTERFACE
  ${DO_Sara_OpenCL_CORE_FILES})

target_include_directories(DO_Sara_OpenCL_Core
  INTERFACE
  ${OpenCL_INCLUDE_DIR}
  $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(DO_Sara_OpenCL_Core
  INTERFACE ${OpenCL_LIBRARIES})

target_compile_definitions(DO_Sara_OpenCL_Core
  INTERFACE
  $<$<PLATFORM_ID:Darwin>:CL_SILENCE_DEPRECATION>
  CL_USE_DEPRECATED_OPENCL_1_0_APIS
  CL_USE_DEPRECATED_OPENCL_1_1_APIS
  CL_USE_DEPRECATED_OPENCL_1_2_APIS)

target_compile_features(DO_Sara_OpenCL_Core
  INTERFACE cxx_std_17)
