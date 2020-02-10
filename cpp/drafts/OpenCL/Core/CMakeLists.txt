file(GLOB DO_Sara_OpenCL_Core_FILES FILES *.hpp)
set_source_files_properties(${DO_Sara_OpenCL_Core} PROPERTIES
  LANGUAGE CXX)

add_custom_target(DO_Sara_OpenCL_Core FILES ${DO_Sara_OpenCL_CORE_FILES})
set_target_properties(DO_Sara_OpenCL_Core PROPERTIES
  FOLDER "Libraries/Sara")
