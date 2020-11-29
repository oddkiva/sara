file(GLOB example_taskflow_SOURCE_FILES FILES *.cpp)

foreach (file ${example_taskflow_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${file})
  target_include_directories(${filename} PRIVATE
    ${Boost_INCLUDE_DIR})
  target_compile_definitions(${filename} PRIVATE
    -DBOOST_ALL_NO_LIB)
endforeach ()
