file(GLOB DO_Sara_OpenCL_Core_FILES FILES *.hpp)
set_source_files_properties(${DO_Sara_OpenCL_Core}
  PROPERTIES
  LANGUAGE CXX)

add_library(DO_Sara_OpenCL_Core INTERFACE)
target_sources(DO_Sara_OpenCL_Core
  INTERFACE
  ${DO_Sara_OpenCL_CORE_FILES})

# set_target_properties(DO_Sara_OpenCL_Core
#   PROPERTIES
#   CXX_STANDARD 17
#   CXX_STANDARD_REQUIRED YES
#   FOLDER "Libraries/Sara")

target_include_directories(DO_Sara_OpenCL_Core
  INTERFACE
  ${OpenCL_INCLUDE_DIR}
  $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries( DO_Sara_OpenCL_Core
  INTERFACE ${OpenCL_LIBRARY})

target_compile_definitions(DO_Sara_OpenCL_Core
  INTERFACE
  CL_USE_DEPRECATED_OPENCL_1_0_APIS
  CL_USE_DEPRECATED_OPENCL_1_1_APIS
  CL_USE_DEPRECATED_OPENCL_1_2_APIS)

target_compile_features(DO_Sara_OpenCL_Core
  INTERFACE cxx_std_17)
