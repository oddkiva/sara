file(GLOB example_taskflow_SOURCE_FILES FILES *.cpp)

foreach (file ${test_Taskflow_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "" "Taskflow")
  target_include_directories(${filename} PRIVATE
    ${Boost_INCLUDE_DIR})
  target_compile_definitions(${filename} PRIVATE
    -DBOOST_ALL_NO_LIB)
endforeach ()
