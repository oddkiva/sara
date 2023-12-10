file(GLOB test_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}
    DEPENDENCIES DO::Sara::Core
    FOLDER KalmanFilter)
  target_compile_options(${filename} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
endforeach ()
