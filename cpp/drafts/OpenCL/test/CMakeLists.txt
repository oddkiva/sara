file(GLOB test_OpenCL_SOURCE_FILES FILES test_*.cpp)

list(APPEND test_deps ${DO_Sara_LIBRARIES} ${OpenCL_LIBRARY} ${Boost_LIBRARIES})
foreach (file ${test_OpenCL_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(${filename} ${file} "${test_deps}" "OpenCL")
  target_include_directories(${filename} PRIVATE ${Boost_INCLUDE_DIR})
  target_compile_definitions(${filename} PRIVATE
    -DBOOST_ALL_NO_LIB
    -DCL_SILENCE_DEPRECATION)
endforeach ()
