file(GLOB test_OpenCL_SOURCE_FILES FILES test_*.cpp)

foreach(file ${test_OpenCL_SOURCE_FILES})
  shakti_add_test(
    SOURCES
    ${file}
    DEPENDENCIES
    DO::Sara::OpenCL
    FOLDER
    OpenCL)
endforeach()
